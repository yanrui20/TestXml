import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from pathlib import Path

def get_new_tb(tb, tb_index, tb_start_index, chan, o_chunks):
    new_tb = deepcopy(tb)
    # 修改chan
    new_tb.set('chan', str(chan)) # 合并的这一份chan都是1
    # 修改id
    new_tb.set('id', str(tb_index))
    # 修改steps
    for step in new_tb.findall('step'):
        # 修改srcoff和dstoff
        for attr in ['srcoff', 'dstoff']:
            srcbuf = step.get("srcbuf")
            if srcbuf == 'o':
                value = int(step.get(attr))
                step.set(attr, str(value + o_chunks * chan))
        # 修改depid
        depid = int(step.get("depid"))
        if depid >= 0:
            step.set("depid", str(depid + tb_start_index))
    return new_tb

def multi_instance(input_file, output_file, instance):
    # 读取XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    nchannels = int(root.get("nchannels"))
    root.set("nchannels", str(nchannels * instance))
    nchunksperloop = int(root.get("nchunksperloop"))
    root.set("nchunksperloop", str(nchunksperloop * instance))

    # 修改所有<gpu>标签的o_chunks属性为32
    for gpu in root.findall('.//gpu'):
        o_chunks = int(gpu.get('o_chunks'))
        gpu.set('o_chunks', str(o_chunks*instance))

        # 复制并处理所有tb标签
        original_tbs = gpu.findall('tb')
        tb_index = len(original_tbs)
        for chan in range(1, instance):
            tb_start_index = tb_index
            for tb in original_tbs:
                new_tb = get_new_tb(tb, tb_index, tb_start_index, chan, o_chunks)
                tb_index += 1
                # 添加到当前GPU节点
                gpu.append(new_tb)

    # 保存修改后的文件
    tree.write(output_file, encoding='UTF-8', xml_declaration=False)

if __name__ == '__main__':
    gpus = 32
    INS = [2, 4, 8, 16]
    xml_dir = f"/Users/yanrui/vscode/nccl/TestXml/Neogen_AG/{gpus}GPUs"
    neogen_dirs = []
    for entry in os.scandir(xml_dir):
        if entry.is_dir() and entry.name.startswith('ring') and "test" not in entry.name:
            neogen_dirs.append(entry.path)
    for input_dir in neogen_dirs:
        for ins in INS:
            output_dir = f"{input_dir}_test_{ins}ins"
            os.makedirs(output_dir, exist_ok=True)
            input = [file for file in Path(input_dir).iterdir() if file.is_file()]
            input = input[0]
            output = [file for file in Path(output_dir).iterdir() if file.is_file()]
            if len(output) > 0:
                output = output[0]
            else:
                output = f"{output_dir}/modified.xml"
            multi_instance(
                input_file=input, 
                output_file=output, 
                instance=ins,
            )