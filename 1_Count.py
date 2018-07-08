# coding: utf-8

# filelist: 
#         train:40428967,  
#     minitrain:4042898,  
# miniminitrain:404291,  
#    test_click:4577464
#  

#import os
#os.chdir('/media/zhou/0004DD1700005FE8/AI/00/project_2/')
#os.chdir('E:/AI/00/project_2')

try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.file_name = 'train'
            self.onehot_name = 'Onehot_B'
            self.data_dir = '../data/project_2/data/{0}/'.format(self.onehot_name)
            self.output_dir = '../data/project_2/output/{0}/'.format(self.onehot_name)
            self.model_dir = '../data/project_2/models/{0}/'.format(self.onehot_name)


#实例化class
FLAGS = flags()


import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss


class Count(object):
    """对特征中的不同取值按一定数据量逐块进行计数, 并保存合并好的统计结果
    
    最终保存文件的格式: 
    columns = ('click=0','click=1','total','ratio')
    index = (column + split + value, 如:'C14=10289')"""
    
    def __init__(self, data_path,):
        """初始化参数"""
        self.data_path = data_path  #数据路径
        #self.file_name = self.data_path + 'train.csv'
        self.file_name = self.data_path + 'train.csv'
        self.data = pd.read_csv(self.file_name, iterator=True)  #迭代读取csv
        self.chunk_size = 1e6  #每次读入数据量
        self.split = '='  # 最终保存文件index的分割符
        self.finale_click_0 = pd.Series()  #对click=0进行计数
        self.finale_click_1 = pd.Series()  #对click=1进行计数
        self.total_size = 0  ##对总的数据量进行计数
        print('counting')
        i = 0
        start_time = time.time()
        while True:
            #i+=1
            #if i>5:break
            try: data_tmp = self.Loaddata()  #每次读取chunk_size条数据 
            except StopIteration: break  #读取结束后跳出循环
            click_0_tmp, click_1_tmp = self.GetClickCount(data_tmp)  #获取统计结果
            self.finale_click_0 = self.CombineSeries(self.finale_click_0, click_0_tmp)
            # 与现有click=0统计结果合并
            self.finale_click_1 = self.CombineSeries(self.finale_click_1, click_1_tmp)
            # 与现有click=1统计结果合并
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            used_time = int(time.time()-start_time)  #记录统计耗时
            print('{0} data have counted, total cost time: {1}'.format(self.total_size, used_time))
            del data_tmp, click_0_tmp, click_1_tmp #及时删除, 以免内存溢出
            gc.collect()
        print('to DataFrame')
        self.ToDataFrame(self.finale_click_0, self.finale_click_1)  
        # 对已经统计好的click=0与click=1统计结果拼接整合, 并转成相应格式的DataFrame
        self.Save()  #保存计数好的文件

    def Loaddata(self, ):
        """每次读取self.chunk_size条数据"""
        return self.data.get_chunk(self.chunk_size)
    
    def ColumnIndex(self, column, index):
        """创建计数文件的索引格式"""
        return column + self.split + str(index)
    
    def GetClickCount(self, tmp):
        """对一定数据量数据进行计数"""
        click_0_index = tmp['click'] == 0
        click_0 = pd.Series()  #初始化click=0
        click_1 = pd.Series()  #初始化click=1
        for column in tmp.columns:  #按列进行计数, 并 逐列 拼接
            if column in ['id','click','hour']: continue
            counts_tmp = self.GetCountSeries(tmp, column, click_0_index)
            # 对当前column, click=0 进行统计, 
            click_0 = pd.concat((click_0, counts_tmp))
            # 并与其column列统计结果 拼接
            counts_tmp = self.GetCountSeries(tmp, column, -click_0_index)
            # 对当前column, click=1 进行统计, 
            click_1 = pd.concat((click_1, counts_tmp))
            # 并与其column列统计结果 拼接
        return click_0, click_1
        
    def GetCountSeries(self, tmp, column, index):
        """对切片好的Series进行频率统计, 并重命名index"""
        counts = tmp[column][index].value_counts()  #
        counts.index = counts.index.map(lambda x:self.ColumnIndex(column,x))
        return counts
        
    def CombineSeries(self, a, b):
        """对传入两Series中相同索引的值累加, 不同索引的值相拼接"""
        same = a.index & b.index
        a[same] += b[same]
        b_unsame = b.index ^ same
        return pd.concat((a, b[b_unsame]))
    
    def ToDataFrame(self, click_0, click_1):
        """将点击数据转为DataFrame"""
        click_0 = click_0.to_frame(name='click=0')  #Series to DataFrame
        click_1 = click_1.to_frame(name='click=1')  #Series to DataFrame
        self.finale_counts = pd.concat((click_0,click_1),axis=1,)#sort=False)
        # 'click=0'与'click=0'进行拼接, 不同索引默认值为 np.nan
        self.finale_counts.fillna(0,inplace=True)  #np.nan to 0
        self.finale_counts['total'] = self.finale_counts['click=0'] + self.finale_counts['click=1']
        # 计算总的频率, 及占总数据量的比例
        self.finale_counts['ratio'] = self.finale_counts['total'] / self.total_size
        #return self.finale_counts
        
    def Save(self, file_name='count.csv'):
        """将数据保存在原有数据文件夹下"""
        print('data saving in {0}'.format(self.data_path+file_name))
        self.finale_counts.to_csv(self.data_path+file_name)


Count(FLAGS.data_dir)