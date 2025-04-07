import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from gen import multi_copy

class PipelineFunc():
    def __init__(self, head_func, tail_func):
        self.is_first_head = head_func
        self.is_first_tail = tail_func

class _tb():
    def __init__(self, tb, gpu_id, func: PipelineFunc):
        self.xml_node = deepcopy(tb)
        self.is_first_head = func.is_first_head(gpu_id, tb)
        self.is_first_tail = func.is_first_tail(gpu_id, tb)

def get_new_wait_step(step_id, depid, deps):
    template = '<step s="0" type="nop" srcbuf="i" srcoff="-1" dstbuf="o" dstoff="-1" cnt="0" depid="8" deps="0" hasdep="0"/>'
    new_step = ET.fromstring(template)
    new_step.set('s', str(step_id))
    new_step.set('depid', str(depid))
    new_step.set('deps', str(deps))
    return new_step

def add_dep_steps(new_tb, wait_steps, start_step_index):
    # 先递增原本的step_id
    dep_num = len(wait_steps)
    # 原本的第一个step可能需要添加最后一个wait step的依赖
    step = new_tb.find(f'step[@s="{start_step_index}"]')
    ori_depid = int(step.get('depid'))
    if ori_depid == -1:
        depid, deps = wait_steps[-1]
        step.set('depid', str(depid))
        step.set('deps', str(deps))
        dep_num -= 1
    # 递增step_id
    for step in new_tb.findall('step'):
        current_s = int(step.get('s'))
        step.set('s', str(current_s + dep_num))
    # 插入新的wait step
    step_id = start_step_index
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
    # 先储存一圈，然后找一下tail个数，tail会被下一个pp的head依赖
    for tb_xml in gpu.findall('tb'):
        tb = _tb(tb_xml, gpu_id, ppfunc)
        if tb.is_first_tail:
            tail_num += 1
        tbs.append(tb)
    # 计算每个pp增加的steps数量
    for tb in tbs:
        tb_id = int(tb.xml_node.get('id'))
        # 首先每轮固定要生成原本那么多
        num_append_steps[tb_id] = len(tb.xml_node.findall('step'))
        if tb.is_first_head:
            num_append_steps[tb_id] += tail_num
            # 原本自身没有依赖，则会减少一个增加的step
            for step in tb.xml_node.findall('step'):
                current_s = int(step.get('s'))
                ori_depid = int(step.get('depid'))
                if current_s == 0 and ori_depid == -1:
                    num_append_steps[tb_id] -= 1
                    break
    return num_append_steps

# 4. 复制stages
def get_new_pipeline_steps(tb: _tb, step_index, o_chunks, pp_index, num_append_steps, tail_steps):
    cur_tb = deepcopy(tb.xml_node)
    # 调整基础数据
    cur_step_index = step_index
    for step in cur_tb.findall('step'):
        # 修改step index
        step.set('s', str(cur_step_index))
        cur_step_index += 1
        # 修改steps的offset属性
        for attr in ['srcoff', 'dstoff']:
            offset = int(step.get(attr))
            if offset != -1:
                step.set(attr, str(offset + o_chunks * pp_index))
        # 修改depid and deps
        depid = int(step.get("depid"))
        deps = int(step.get("deps"))
        if depid >= 0: # 有依赖
            # 这里是依赖于当前pp, pp_index不为0的时候, 依赖的deps已经发生了变化
            deps += num_append_steps[depid] * pp_index
            step.set("deps", str(deps))
    # 增加新的依赖steps
    if tb.is_first_head and pp_index != 0:
        wait_steps = tail_steps.copy()
        # 处理tails
        # 处理num_append_steps
        for i in range(len(wait_steps)):
            depid, deps = wait_steps[i]
            # 这里依赖的是上一个pp，如果当前pp_index==1，则上一个stage（pp=0）的deps并没有变化
            if pp_index > 1:
                deps += num_append_steps[depid] * (pp_index - 1)
            wait_steps[i] = (depid, deps)
        cur_tb = add_dep_steps(cur_tb, wait_steps, step_index)

    return cur_tb.findall('step')

def get_new_pipeline_tbs(tb_index, pp_index, gpu_info):
    new_tb = deepcopy(gpu_info.tbs[tb_index].xml_node)
    o_chunks = gpu_info.o_chunks
    # 修改chan和id
    new_tb.set('chan', str(pp_index))
    new_tb.set('id', str(tb_index + gpu_info.num_tbs * pp_index))
    # 修改steps
    for step in new_tb.findall('step'):
        # 修改srcoff和dstoff
        for attr in ['srcoff', 'dstoff']:
            offset = int(step.get(attr))
            if offset != -1:
                step.set(attr, str(offset + o_chunks * pp_index))
        # 修改depid
        depid = int(step.get("depid"))
        if depid >= 0:
            depid += gpu_info.num_tbs * pp_index
            step.set("depid", str(depid))
    # 增加依赖的step
    if gpu_info.tbs[tb_index].is_first_head and pp_index != 0:
        wait_steps = gpu_info.tail_steps.copy()
        # 处理num_append_steps
        for i in range(len(wait_steps)):
            depid, deps = wait_steps[i]
            depid += gpu_info.num_tbs * (pp_index - 1)
            wait_steps[i] = (depid, deps)
            # 还需要让被依赖的step的hasdep=1
            dep_step = gpu_info.gpu.find(f'.//tb[@id="{depid}"]/step[@s="{deps}"]')
            dep_step.set('hasdep', '1')
        new_tb = add_dep_steps(new_tb, wait_steps, 0)
    return new_tb

class GpuInfo():
    def __init__(self, gpu, ppfunc):
        self.gpu = gpu
        self.o_chunks = int(gpu.get('o_chunks'))
        self.gpu_id = int(gpu.get('id'))
        self.tbs = [] # 复制所有tb的信息
        self.num_tbs = len(gpu.findall('tb'))
        self.tail_steps = {} # 记录第一个阶段每个的tail tb的最后一个 step（一般来说只有一个）
        # self.increase_depid = {} # 维护一个全局的depid增长对应关系
        # self.num_append_steps = how_many_steps_need_append(gpu)
        for tb_xml in gpu.findall('tb'):
            # 判断是否是第一个stage，以及是否是head和tail
            tb = _tb(tb_xml, self.gpu_id, ppfunc)
            self.tbs.append(tb)
            if tb.is_first_tail:
                tb_id = int(tb.xml_node.get('id'))
                last_step_id = len(tb.xml_node.findall('step')) - 1
                self.tail_steps = [(tb_id, last_step_id)]
            # # 维护一个全局的depid增长关系
            # tb_id = int(tb.xml_node.get('id'))
            # self.increase_depid[tb_id] = [tb_id]

def multi_pipeline(input_file, output_file, pipeline, ppfunc):
    # 读取XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    # 1. 处理root的nchunksperloop和nchannels
    nchunksperloop = int(root.get("nchunksperloop"))
    root.set("nchunksperloop", str(nchunksperloop * pipeline))
    nchannels = int(root.get("nchannels"))
    root.set("nchannels", str(nchannels * pipeline))
    root.set("outofplace", str(1))
    for gpu in root.findall('.//gpu'):
        # 2. 复制GPU信息
        gpu_info = GpuInfo(gpu, ppfunc)
        # 3. 要处理gpu标签的o_chunks属性
        gpu.set('o_chunks', str(gpu_info.o_chunks*pipeline))
        # 4. 复制tb
        for pp_index in range(1, pipeline):
            for tb_index in range(len(gpu_info.tbs)):
                new_tb = get_new_pipeline_tbs(tb_index, pp_index, gpu_info)
                gpu.append(new_tb)
    # 格式化, 2个空格缩进
    ET.indent(tree, space='  ', level=0)
    # 保存修改后的文件
    tree.write(output_file, encoding='UTF-8', xml_declaration=False)


# 3. 是否是第一个stage head
def is_first_head_ring_8_4(gpu_id, tb_xml):
    send = int(tb_xml.get('send'))
    recv = int(tb_xml.get('recv'))
    if recv == -1 and send >= 0 and (send // 8) != (gpu_id // 8): # send不在同一个8卡中
        return True
    return False

# 3. 是否是第一个stage tail
def is_first_tail_ring_8_4(gpu_id, tb_xml):
    send = int(tb_xml.get('send'))
    recv = int(tb_xml.get('recv'))
    if send == -1 and recv >= 0 and (recv // 8) != (gpu_id // 8): # recv不在同一个8卡中
        return True
    return False

if __name__ == '__main__':
    ppfunc = PipelineFunc(
        head_func=is_first_head_ring_8_4,
        tail_func=is_first_tail_ring_8_4,
    )
    input = "./Neogen_AG/32GPUs/ring8_4/fullmesh_2hosts_32nodes_8_4.txt.xml"
    for pipeline in [1, 2, 4, 8, 16]:
        for ins in [1, 2, 4, 8]:
            if pipeline * ins > 64:
                continue
            output = f"./Neogen_AG/16GPUs_pipeline/mesh_8_2_multi_chan_pp_{pipeline}_ins_{ins}/test.xml"
            os.makedirs(os.path.dirname(output), exist_ok=True)
            multi_pipeline(input, output, pipeline, ppfunc)
            multi_copy(output, output, ins)
