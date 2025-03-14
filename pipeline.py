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

def get_new_wait_step(step_id, depid, deps):
    template = '<step s="0" type="nop" srcbuf="i" srcoff="-1" dstbuf="o" dstoff="-1" cnt="0" depid="8" deps="0" hasdep="0"/>'
    new_step = ET.fromstring(template)
    new_step.set('s', str(step_id))
    new_step.set('depid', str(depid))
    new_step.set('deps', str(deps))
    return new_step

# 4. 复制stages
def get_new_pipeline_tb(tb: _tb, tb_index, tb_start_index, last_tb_start_index, o_chunks, pp_index, wait_steps, head_depids):
    new_tb = deepcopy(tb.xml_node)
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
    # 如果是head，需要等待上一个stage的tail(head原本是没有依赖的，不需要修改原本的依赖)
    if tb.is_first_head:
        # 先递增原本的step_id
        for step in new_tb.findall('step'):
            current_s = int(step.get('s'))
            step.set('s', str(current_s + len(wait_steps) - 1))
            # 原本的第一个step需要添加最后一个wait step的依赖
            if current_s == 0:
                depid, deps = wait_steps[-1]
                depid += last_tb_start_index
                step.set('depid', str(depid))
                step.set('deps', str(deps))
        # 插入新的wait step
        step_id = 0
        for wait_step in wait_steps[:-1]:
            depid, deps = wait_step
            depid += last_tb_start_index
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
    else: # 不是head
        for step in new_tb.findall('step'):
            depid = int(step.get('depid'))
            deps = int(step.get('deps'))
            # 原本的deps是否需要修改, 即depid是否是head，是则需要增加len(wait_steps)
            if depid in head_depids:
                step.set('deps', str(deps + len(wait_steps)))
            # 原有的depid只需叠加index即可
            if depid >= 0:
                step.set('depid', str(depid + tb_start_index))
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
        wait_steps = []
        head_depids = []
        for tb_xml in original_tbs:
            # 判断是否是第一个stage，以及是否是head和tail
            tb = _tb(tb_xml, gpu_id, ppfunc)
            if tb.is_first_tail:
                tb_id = int(tb.xml_node.get('id'))
                last_step_id = len(tb.xml_node.findall('step'))
                wait_steps.append((tb_id, last_step_id))
            if tb.is_first_head:
                tb_id = int(tb.xml_node.get('id'))
                head_depids.append(tb_id)
            tbs.append(tb)
        # 4. 复制stage
        tb_start_index = 0
        tb_index = len(original_tbs)
        for pp_index in range(1, pipeline):
            last_tb_start_index = tb_start_index
            tb_start_index = tb_index
            for tb in tbs:
                new_tb_xml = get_new_pipeline_tb(tb, tb_index, tb_start_index, last_tb_start_index, o_chunks, pp_index, wait_steps, head_depids)
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
    for pipeline in [4]: # 1, 2, 4, 8, 16, 32
        for instance in [1]: # 1, 2, 4, 8
            output = f"./Neogen_AG/32GPUs_pipeline/mesh_8_4_pp_{pipeline}_ins_{instance}/test.xml"
            os.makedirs(os.path.dirname(output), exist_ok=True)
            multi_pipeline(input, output, pipeline, ppfunc)
            multi_instance(output, output, instance)
