import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from gen import multi_instance

class PipelineFunc():
    def __init__(self, head_func, tail_func):
        self.is_first_head = head_func
        self.is_first_tail = tail_func

class _tb():
    def __init__(self, tb, gpu_id, func: PipelineFunc):
        self.xml_node = tb
        self.is_first_head = func.is_first_head(gpu_id, tb)
        self.is_first_tail = func.is_first_tail(gpu_id, tb)
        self.is_recv = (int(tb.get('recv')) != -1)

def get_new_wait_step(step_id, depid, deps):
    template = '<step s="0" type="nop" srcbuf="i" srcoff="-1" dstbuf="o" dstoff="-1" cnt="0" depid="8" deps="0" hasdep="0"/>'
    new_step = ET.fromstring(template)
    new_step.set('s', str(step_id))
    new_step.set('depid', str(depid))
    new_step.set('deps', str(deps))
    return new_step

def add_dep_steps(new_tb, wait_steps):
    # 先递增原本的step_id
    dep_num = len(wait_steps)
    for step in new_tb.findall('step'):
        current_s = int(step.get('s'))
        ori_depid = int(step.get('depid'))
        # 原本的第一个step可能需要添加最后一个wait step的依赖
        if current_s == 0 and ori_depid == -1:
            depid, deps = wait_steps[-1]
            step.set('depid', str(depid))
            step.set('deps', str(deps))
            dep_num -= 1
        step.set('s', str(current_s + dep_num))
    # 插入新的wait step
    step_id = 0
    for wait_step in wait_steps[:dep_num]:
        depid, deps = wait_step
        new_step = get_new_wait_step(step_id, depid, deps)
        step_id += 1
        new_tb.append(new_step)
    # 排序step节点
    steps = new_tb.findall('step')
    steps.sort(key=lambda x: int(x.get('s')))
    # 删除原本的step节点
    del new_tb[:]
    # 插入排序好的step节点
    for step in steps:
        new_tb.append(step)
    return new_tb

def how_many_steps_need_append(gpu):
    num_append_steps = {}
    gpu_id = int(gpu.get('id'))
    tbs = []
    tail_num = 0
    for tb_xml in gpu.findall('tb'):
        tb = _tb(tb_xml, gpu_id, ppfunc)
        if tb.is_first_tail:
            tail_num += 1
        tbs.append(tb)
    for tb in tbs:
        tb_id = int(tb.xml_node.get('id'))
        if tb.is_first_head:
            num_append_steps[tb_id] = tail_num
            # 原本自身没有依赖，则会减少一个增加的step
            for step in tb.xml_node.findall('step'):
                current_s = int(step.get('s'))
                ori_depid = int(step.get('depid'))
                if current_s == 0 and ori_depid == -1:
                    num_append_steps[tb_id] -= 1
                    break
        elif tb.is_recv:
            num_append_steps[tb_id] = 1
            # 原本自身没有依赖，则会减少一个增加的step
            for step in tb.xml_node.findall('step'):
                current_s = int(step.get('s'))
                ori_depid = int(step.get('depid'))
                if current_s == 0 and ori_depid == -1:
                    num_append_steps[tb_id] -= 1
                    break
        else:
            num_append_steps[tb_id] = 0
    return num_append_steps

# 4. 复制stages
def get_new_pipeline_tb(tb: _tb, tb_index, tb_start_index, last_tb_start_index, o_chunks, pp_index, is_last_pp, tail_steps, num_append_steps):
    new_tb = deepcopy(tb.xml_node)
    # 修改chan
    new_tb.set('chan', str(pp_index))
    # 修改id
    new_tb.set('id', str(tb_index))
    # 修改steps的offset属性
    for step in new_tb.findall('step'):
        # 修改srcoff和dstoff
        for attr in ['srcoff', 'dstoff']:
            srcbuf = step.get("srcbuf")
            if srcbuf == 'o':
                value = int(step.get(attr))
                step.set(attr, str(value + o_chunks * pp_index))
    # 修改step依赖
    # 修改depid and deps
    for step in new_tb.findall('step'):
        depid = int(step.get("depid"))
        deps = int(step.get("deps"))
        if depid >= 0:
            if pp_index != 0: # 这里是依赖于当前stage, pp_index不为0的时候, 依赖的deps已经发生了变化
                deps += num_append_steps[depid]
            step.set("deps", str(deps))
            step.set("depid", str(depid + tb_start_index))
    # 添加hasdep
    if tb.is_recv and not is_last_pp:
        for step in new_tb.findall('step'):
            step.set("hasdep", "1")
    # 需要制作wait steps
    if tb.is_first_head and pp_index != 0:
        # 如果是head, 需要等待上一个stage的tail
        wait_steps = tail_steps.copy()
        for i in range(len(wait_steps)):
            depid, deps = wait_steps[i]
            # 这里依赖的是上一个stage，如果当前pp_index==1，则上一个stage（pp=0）的deps并没有变化
            if pp_index > 1:
                deps += num_append_steps[depid]
            wait_steps[i] = (depid + last_tb_start_index, deps)
        new_tb = add_dep_steps(new_tb, wait_steps)
    elif tb.is_recv and pp_index != 0:
        # 如果是recv, 需要等待上一个stage的对应的recv
        tb_id = int(tb.xml_node.get('id'))
        deps = len(new_tb.findall('step')) - 1
        if pp_index > 1:
            deps += num_append_steps[tb_id]
        wait_steps = [(tb_id + last_tb_start_index, deps)]
        new_tb = add_dep_steps(new_tb, wait_steps)
    return new_tb

# 3. 是否是第一个stage head
def is_first_head_mesh_8_4(gpu_id, tb_xml):
    send = int(tb_xml.get('send'))
    recv = int(tb_xml.get('recv'))
    if gpu_id == 0:
        if recv == -1 and (send // 8) != (gpu_id // 8): # send不在同一个8卡中
            return True
    elif send == recv == -1: ## 奇怪的copy
        return True
    return False

# 3. 是否是第一个stage tail
def is_first_tail_mesh_8_4(gpu_id, tb_xml):
    send = int(tb_xml.get('send'))
    recv = int(tb_xml.get('recv'))
    if send == -1 and recv > 0 and (recv // 8) != (gpu_id // 8): # recv不在同一个8卡中
        return True
    return False

def multi_pipeline(input_file, output_file, pipeline, ppfunc):
    # 读取XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    # 1. 处理root的nchunksperloop
    nchunksperloop = int(root.get("nchunksperloop"))
    root.set("nchunksperloop", str(nchunksperloop * pipeline))
    for gpu in root.findall('.//gpu'):
        # 2. 要处理gpu标签的o_chunks
        o_chunks = int(gpu.get('o_chunks'))
        gpu.set('o_chunks', str(o_chunks*pipeline))
        # 复制并处理所有tb标签
        gpu_id = int(gpu.get('id'))
        original_tbs = gpu.findall('tb')
        tbs = []
        tail_steps = []
        for tb_xml in original_tbs:
            # 判断是否是第一个stage，以及是否是head和tail
            tb = _tb(tb_xml, gpu_id, ppfunc)
            if tb.is_first_tail:
                tb_id = int(tb.xml_node.get('id'))
                last_step_id = len(tb.xml_node.findall('step'))
                tail_steps.append((tb_id, last_step_id))
            tbs.append(tb)
        # 4. 复制stage
        tb_start_index = 0
        tb_index = 0
        num_append_steps = how_many_steps_need_append(gpu)
        del gpu[:]
        for pp_index in range(pipeline):
            last_tb_start_index = tb_start_index
            tb_start_index = tb_index
            is_last_pp = (pp_index == pipeline - 1)
            for tb in tbs:
                new_tb_xml = get_new_pipeline_tb(tb, tb_index, tb_start_index, last_tb_start_index, o_chunks, pp_index, is_last_pp, tail_steps, num_append_steps)
                tb_index += 1
                gpu.append(new_tb_xml)
    # 格式化, 2个空格缩进
    ET.indent(tree, space='  ', level=0)
    # 保存修改后的文件
    tree.write(output_file, encoding='UTF-8', xml_declaration=False)

if __name__ == '__main__':
    # multi_instance(input, output, instance)
    ppfunc = PipelineFunc(
        head_func=is_first_head_mesh_8_4,
        tail_func=is_first_tail_mesh_8_4,
    )
    input = "./Neogen_AG/32GPUs/ring8_4/fullmesh_2hosts_32nodes_8_4.txt.xml"
    for pipeline in [2, 4]: # 1, 2, 4, 8, 16, 32
        for instance in [1]: # 1, 2, 4, 8
            output = f"./Neogen_AG/32GPUs_pipeline/mesh_8_4_pp_{pipeline}_ins_{instance}/test.xml"
            os.makedirs(os.path.dirname(output), exist_ok=True)
            multi_pipeline(input, output, pipeline, ppfunc)
            multi_instance(output, output, instance)
