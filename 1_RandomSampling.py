import pandas as pd
import numpy as np
import time
import gc


def RandomSampling(file_name, output_name, mini_size=0.1, target='click', chunk_size=1e6):
    """随机对数据进行采样"""    
    
    tmp = pd.read_csv(file_name, iterator=True)  #迭代读取书据
    tmp.get_chunk(1).to_csv(output_name, mode='w', index=False)  
    #提前写入一条数据,带有表头, 以免下次写入时表头重复
    size = np.array([[0,0]])  #统计数据大小
    i=0
    while True:
        #i+=1
        #if i>10:break
        try: data = tmp.get_chunk(chunk_size)  #每次读取chunk_size条数据 
        except StopIteration: break  #读取结束后跳出循环
        mini_data = data.sample(frac=mini_size, random_state=33, ) 
        # 随机抽取mini_size(占总数据的比例)条数据
        size = np.concatenate((size,[[data.shape[0], mini_data.shape[0]]]), axis=0)
        mini_data.to_csv(output_name, mode='a', header=False, index=False)
        # 保存抽取后的数据, 不要表头(否则每次保存均有表头), 不要索引(否则单独开启一列保存索引)
        del data, mini_data #及时删除, 以免内存溢出
        gc.collect()
    size = size.sum(axis=0)
    print('file size is {0}, mini size is {1}'.format(size[0],size[1]))"
