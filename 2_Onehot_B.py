import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
import xgboost as xgb
import tarfile
from multiprocessing import Pool

from utils import *
from params import *


def Onehot_A( param ):
    """
    对传入数据进行onehot编码, 转成scipy.sparse的稀疏矩阵, 并保存为.npz
    
    获取参数字典, 由于多进程Pool.map只传入一个参数
    """
    data_path = param['data_path']  #数据路径
    file_name = param['file_name']  #待处理文件名
    chunksize = param['chunksize']  #迭代读取数据量
    data_begin = param['data_begin']  #数据起始索引
    data_end = param['data_end']  #数据终止索引
    output_path = param['output_path']  #输出文件夹
    threshold = param['threshold']  #对频率>=threshold的特征, 进行onehot编码
    model_path = param['model_path']
    num_trees = param['num_trees']
    deep = param['deep']
    onehot_name = param['onehot_name']
    lr_C = param['lr_C']

    # 实例化各类工具
    data_utils = DataUtils()
    onehot_utils = OnehotUtils()
    npz_utils = NpzUtils()

    # 迭代导入数据
    data = data_utils.LoadData(data_path, file_name)  
    # 跳过前data_begin条数据
    data = data_utils.JumpData(data, data_begin, chunksize)  


    if onehot_name == 'A_cat':
        Onehot_Encoder = onehot_utils.GetOneHot_Cat
        # 获取onehot的columns
        get_index = onehot_utils.GetCategoryColums(data_path, 'count', threshold)  
    elif onehot_name == 'A_hour':
        Onehot_Encoder = onehot_utils.GetOneHot_Hour
        # 获取onehot的columns
        get_index = None
    elif onehot_name == 'A_xgb':
        X_train, y_train = onehot_utils.GetOneHot_XGB(output_path, file_name, model_path, 
                                    data_begin, num_trees, deep, onehot_name )
        npz_utils.SaveNpz(X_train, y_train, output_path, 
                            file_name, data_begin, onehot_name)  #保存 稀疏矩阵.npz
        return None
    elif onehot_name == 'B_cat':
        Onehot_Encoder = onehot_utils.GetOneHot_Cat
        # 获取onehot的columns
        get_index = onehot_utils.GetCategoryColums(data_path, 'count', threshold)  
    elif onehot_name == 'A_his':
        Onehot_Encoder = onehot_utils.GetOneHot_His
        # 获取onehot的columns
        get_index = onehot_utils.MergeAllClickHistory(data_path,)  
    
    start_time = time.time()
    
    # 每次读取chunk_size条数据 
    data_tmp = data_utils.NextChunk(data, chunksize)  
    # 生成第一次数据的稀疏矩阵, 为后面的拼接做准备
    X_train, y_train = Onehot_Encoder(data_tmp,  get_index,) 
    #used_time = int(time.time()-start_time)
    #print('{0} data have to OneHot, total cost time: {1}'.format(self.total_size, used_time))

    #对之后数据进行onehot编码
    while True:
        if data_utils.total_size >= data_end: break  #先判断数据量是否足够, 若足够, 训练结束
        data_tmp = data_utils.NextChunk(data, chunksize)  #每次读取chunk_size条数据 
        if data_utils.data_over: break #数据读取结束, 训练结束
        X_train_tmp, y_train_tmp = Onehot_Encoder(data_tmp,  get_index,) 
        X_train = ss.vstack((X_train, X_train_tmp))  #与原有 行稀疏矩阵 进行 行连接
        y_train = ss.hstack((y_train, y_train_tmp))  #与原有 列稀疏矩阵 进行 列连接
        if data_utils.total_size % (chunksize*10) ==0:
            used_time = int(time.time()-start_time)  #记录统计耗时
            print('{0} data have to OneHot, total cost time: {1}'.format(data_utils.total_size, used_time))
        #del X_train_tmp, y_train_tmp, data_tmp #及时删除, 以免内存溢出
        gc.collect()
    #return X_train, y_train

    npz_utils.SaveNpz(X_train, y_train, output_path, file_name, data_begin, onehot_name)  #保存 稀疏矩阵.npz


# ===========================================
# filelist: 
#         train:40428967,   count:9449445,
#     minitrain:4042898,   more_5:1544466
# miniminitrain:404291,   more_10:645381
#    test_click:4577464
#
#   Onehot_A: ['A_cat', 'A_hour', 'A_xgb', 'A_his']
#   Onehot_B: ['B_cat', ]
# ===========================================

if __name__ == "__main__":

    FLAGS = flags('minitrain', 'Onehot_B')
    PARAMS = params('B_cat')    
    # 解压提前压缩好的tar.gz文件, 
    # 主要用于解压上传到tinymind中的数据, 
    # 本地运行不需要解压 
    DataUtils().ExtractData(FLAGS.data_dir)

    # 设定参数
    totalsize = {'train':40428967, 
                'minitrain':4042898, 
                'test_click':4577464}
    # 总的数据量
    file_size = totalsize[FLAGS.file_name]  
    block_size = 100000  #数据块大小

    param =[dict( 
            data_path = FLAGS.data_dir,
            file_name = FLAGS.file_name,
            output_path = FLAGS.output_dir,
            model_path = FLAGS.model_dir,
            #每次处理数据的多少, 必须被block_size整除
            chunksize = PARAMS.chunksize,  
            data_end = begin+block_size,
            threshold = PARAMS.threshold,
            num_trees = PARAMS.num_trees,
            deep = PARAMS.deep,
            onehot_name = PARAMS.onehot_name,
            lr_C = PARAMS.lr_C,
            data_begin = begin, 
                ) 
            for begin in range(0,file_size,block_size)
            ]

    # 多进程处理onehot
    Onehot_A(param[1])

    # 4为进程数, 可改为更大
    with Pool(4) as p:  
        p.map(Onehot_A, param)

    NpzUtils().MergeNpz(FLAGS.output_dir, 
                        FLAGS.file_name, 
                        [begin for begin in range(0,file_size,block_size)], 
                        PARAMS.onehot_name,)

###############################################################

    FLAGS = flags('test_click', 'Onehot_B')
    PARAMS = params('B_cat')    
    # 解压提前压缩好的tar.gz文件, 
    # 主要用于解压上传到tinymind中的数据, 
    # 本地运行不需要解压 
    DataUtils().ExtractData(FLAGS.data_dir)

    # 设定参数
    totalsize = {'train':40428967, 
                'minitrain':4042898, 
                'test_click':4577464}
    # 总的数据量
    file_size = totalsize[FLAGS.file_name]  
    block_size = 100000  #数据块大小

    param =[dict( 
            data_path = FLAGS.data_dir,
            file_name = FLAGS.file_name,
            output_path = FLAGS.output_dir,
            model_path = FLAGS.model_dir,
            #每次处理数据的多少, 必须被block_size整除
            chunksize = PARAMS.chunksize,  
            data_end = begin+block_size,
            threshold = PARAMS.threshold,
            num_trees = PARAMS.num_trees,
            deep = PARAMS.deep,
            onehot_name = PARAMS.onehot_name,
            lr_C = PARAMS.lr_C,
            data_begin = begin, 
                ) 
            for begin in range(0,file_size,block_size)
            ]

    # 多进程处理onehot
    Onehot_A(param[1])

    # 4为进程数, 可改为更大
    with Pool(4) as p:  
        p.map(Onehot_A, param)

    NpzUtils().MergeNpz(FLAGS.output_dir, 
                        FLAGS.file_name, 
                        [begin for begin in range(0,file_size,block_size)], 
                        PARAMS.onehot_name,)
