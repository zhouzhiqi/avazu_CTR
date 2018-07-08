
import os
print(os.getcwd())
#os.chdir('/media/zhou/0004DD1700005FE8/AI/00/project_2/')
os.chdir('E:/AI/00/project_2')
print(os.getcwd())


try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.file_name = 'minitrain'
            self.output_dir = '../data/project_2/models/'
            self.data_dir = '../data/project_2/output_{0}/'.format(self.file_name)
            self.model_dir = '../data/project_2/models/'
            self.chunksize = 1e3
            self.threshold = 10
            self.data_begin = 0
            self.data_end = 1e5
            self.id_index = 0
            self.num_trees = 30
            self.max_depth = 8

#实例化class
FLAGS = flags()

import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
import xgboost as xgb
from sklearn.externals import joblib


data_path = FLAGS.data_dir  
file_name = FLAGS.file_name
chunksize = FLAGS.chunksize
threshold = FLAGS.threshold
output_path = FLAGS.output_dir
deep = FLAGS.max_depth + 1   #实际树的深度为 max_depth+1
num_trees = FLAGS.num_trees  #树的数量
model_path = FLAGS.model_dir

#保存文件中的'id'
class SaveID(object):
    """把文件中的'id'列读取出来, 并保存"""

    def __init__(self, data_path, output_path, file_name, chunksize, index=0):
        self.data_over = False
        chunksize *= 1000 #每次读取数据
        output_name = output_path + '{0}_id.csv'.format(file_name)  #输出文件的名子
        data = self.LoadData(data_path, file_name, index)  #迭代导入数据
        data.get_chunk(1).to_csv(output_name, mode='w', index=False)  #先写入一条数据, 带有header
        self.total_size = 1
        while True:
            data_tmp = self.NextChunk(data, chunksize)  #迭代读取数据
            if self.data_over: break  #如果数据读取完毕, 跳出循环
            data_tmp.to_csv(output_name, mode='a', header=False, index=False) #累加保存csv文件
            gc.collect()
        print('the size of [id] in {0}.csv is: {1} '.format(file_name, self.total_size))

    def LoadData(self, data_path, file_name, index):
        """迭代加载数据"""
        data = pd.read_csv(data_path+'{0}.csv'.format(file_name), 
                            usecols=[index], dtype=np.uint64, iterator=True)  #[index]是'id'所在的列索引
        return data

    def NextChunk(self, data, chunksize):
        """获取 chunksize 条数据"""
        #0.4M条数据用时44min, 改变chunksize基本不会影响处理速度
        try: data_tmp = data.get_chunk(chunksize)  #每次读取chunk_size条数据 
        except StopIteration: self.data_over = True  #读取结束后跳出循环
        else: 
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            return data_tmp  #返回取出的数据
