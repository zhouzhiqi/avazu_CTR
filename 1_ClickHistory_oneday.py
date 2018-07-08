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
            self.file_name = 'minitrain'
            self.onehot_name = 'Onehot_A'
            self.data_dir = '../data/project_2/data/{0}/'.format(self.onehot_name)
            self.output_dir = '../data/project_2/output/{0}/'.format(self.onehot_name)
            self.model_dir = '../data/project_2/models/{0}/'.format(self.onehot_name)

class params(object):
    def __init__(self, ):
        self.threshold = 10
        self.chunksize = 1e4
        self.num_trees = 50
        self.deep = 8
        self.split = '='

#实例化class
FLAGS = flags()
PARAMS = params()

import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
#from  sklearn.cross_validation  import  train_test_split 
#import xgboost as xgb
import tarfile
from multiprocessing import Pool
import pickle as pickle

#解压文件
def ExtractData( data_path):
    """把data_path中的tar.gz文件全部解压出来"""
    file_list = os.listdir(data_path)  #文件列表
    if 'count.csv' in file_list: return None  #若已经解压, pass
    for file_name in file_list: #把所有.tar.gz解压到data_path
        with tarfile.open(data_path + file_name, 'r:gz') as tar: 
            tar.extractall(data_path)


def GetC17IIndex(data_path, ):
    print('get C17s ')
    C17 = pd.read_csv(data_path+'train.csv', usecols=[19],)
    C17s = C17.loc[:,'C17'].value_counts()
    C17s = C17s.index.map(lambda x : 'C17=' + str(x))
    #C17_index = pd.Series(index=C17s, data=np.arange(4,435+4))
    del C17
    gc.collect()
    return C17s

class ClickHistory(object):
    
    def __init__(self, param, ):
        data_path = param['data_path']
        file_name = param['file_name']
        chunksize = param['chunksize']
        output_path = param['output_path']
        day = param['day']
        #C17s = param['C17s']
        self.total_size = 0
        self.data_over = False
        self.one_day = self.GetOneDay(data_path, file_name, chunksize, day)
        print('get one day')
        C17s = None
        self.click_history = self.GetHistory(self.one_day, day, C17s)
        self.click_history.to_csv(data_path+'click_history_day{0}.csv'.format(day), index=False)

    def LoadData(self, data_path, file_name):
        print('load data')
        data = pd.read_csv(data_path+'{0}.csv'.format(file_name), 
                           usecols=[1,2,3,4,11,12,13,19],
                           iterator=True)
        return data

    def NextChunk(self, data, chunksize):
        """获取 chunksize 条数据"""
        #0.4M条数据用时44min, 改变chunksize基本不会影响处理速度
        try: data_tmp = data.get_chunk(chunksize)  #每次读取chunk_size条数据 
        except StopIteration: self.data_over = True  #读取结束后跳出循环
        else: 
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            return data_tmp  #返回取出的数据

    def GetC17IIndex(self, data_path, ):
        C17 = pd.read_csv(data_path+'train.csv', usecols=[19],)
        C17s = C17.loc[:,'C17'].value_counts()
        C17s = C17s.index.map(lambda x : 'C17=' + str(x))
        #C17_index = pd.Series(index=C17s, data=np.arange(4,435+4))
        del C17
        gc.collect()
        return C17s

    def DayHourUserC17(self, data, day):
        columns = ['click', 'hour', 'day', 'user', 'device_id', 'device_ip', 'device_model', 'C17']
        data.columns = columns
        data.loc[:,'day'] = data.loc[:, 'hour'].values %14100000 //100
        if day not in data.loc[:,'day'].values: 
            return pd.DataFrame(columns=columns,)
        data.loc[:,'hour'] = data.loc[:, 'hour'].values %14100000 %100
        #data.loc[:, 'C17'] = 'C17=' + data.loc[:, 'C17'].astype(np.str)

        data.loc[:,'user'] = data.loc[:, 'device_id'].values
        index = data.loc[:, 'device_id']=='a99f214a'
        data.loc[index, 'user'] = data.loc[index, 'device_ip']  + '&' + data.loc[index, 'device_model']

        return data.loc[:,['click', 'hour', 'day', 'user', 'C17']]

    def MergeOneDay(self, data, day):
        index = data.loc[:,'day'] == day
        return data.loc[index, :]

    def GetOneDay(self, data_path, file_name, chunksize, day,):
        print('getting day:', day)
        strat_time = time.time()
        data = self.LoadData(data_path, file_name,)
        data_tmp = self.NextChunk(data, chunksize)
        data_tmp = self.DayHourUserC17(data_tmp, day)
        Oneday = self.MergeOneDay(data_tmp, day)
        while True:
            #if self.total_size >= 1e6: break  #先判断数据量是否足够, 若足够, 训练结束
            data_tmp = self.NextChunk(data, chunksize)  #每次读取chunk_size条数据 
            if self.data_over: break #数据读取结束, 训练结束
            data_tmp = self.DayHourUserC17(data_tmp, day)
            Oneday = pd.concat((Oneday, self.MergeOneDay(data_tmp, day)),axis=0)
            #print(Oneday.shape)
        print('getting a day cost time:', int(time.time() - strat_time))
        return Oneday
    
    def GetHistory(self, data, day, C17s):
        print('getting click history')
        columns = pd.Index(['day', 'user','click_times', 'total_times', 'C17_Fir', 'C17_Sec'])#.append(C17s)
        click_history = pd.DataFrame(columns=columns,)#data=np.zeros((1,columns.shape[0]),dtype=np.uint16))
        data.index = data.loc[:,'user']
        users = data.loc[:,'user'].value_counts().index

        data.loc[:,'user'] = 0
        total_size = users.shape[0]
        print('total size:', total_size)
        block_size = 1000
        start_time = time.time()
        for i in np.arange(0,total_size,block_size):
            user_tmp = users[i:i+block_size]
            data_tmp = data.loc[user_tmp, :]
            click_day = pd.DataFrame(columns=columns, index=user_tmp, data=np.zeros((user_tmp.shape[0], columns.shape[0]), dtype=np.int32))
            for u in user_tmp: 
                data_oneuser = data_tmp.loc[u,:]
                click_day.loc[u,:] = self.GetOneUser(data_oneuser, 666, 666, u, )
            click_day.loc[:,'day'] = day
            click_day.loc[:,'user'] = user_tmp
            click_history = pd.concat((click_history,click_day), axis=0)
            if  i%10000==0:
                used_time = int(time.time()-start_time)
                print('day[{0}] in size[{1:0.3f}], total time:{2}'.format(day, (i+block_size)/total_size, used_time),)
            #break
        return click_history

    def GetOneUser(self, data_user, day, hour, user, ):
        #columns = pd.Index(['day', 'user','click_times', 'total_times', 'C17_Fir', 'C17_Sec'])
        data_tmp = np.zeros((6),dtype=np.int32)
        #data_tmp[0] = hour
        #data_tmp[1] = user
        if len(data_user.shape)==1: 
            data_tmp[2]=data_user['click']
            data_tmp[3] = 1
            data_tmp[-2] = data_user['C17']
            return data_tmp

        data_tmp[2] = data_user.loc[:,'click'].sum()
        data_tmp[3] = data_user.shape[0]
        C17_count = data_user.loc[:,'C17'].value_counts().index 
        data_tmp[-2] = C17_count[0]
        try: data_tmp[-1] = C17_count[1]
        except IndexError: pass
        return data_tmp


def MergeAllClickHistoryMax( param, ):
    data_path = param['data_path']
    days = [d for d in range(21,31)]
    for d in days:
        print(d, end='   ')
        data_tmp = pd.read_csv(data_path+'click_history_{0}.csv'.format(d))
        day_tmp = data_tmp.pop('day')
        user_tmp = data_tmp.pop('user')
        print(max(data_tmp.max()))
    

    
def ToPickleDump(param,):
    data_path = param['data_path']
    day = param['day']
    data_tmp = pd.read_csv(data_path+'click_history_day{0}.csv'.format(day), usecols=np.arange(2,439), dtype=np.uint16)
    data_index = pd.read_csv(data_path+'click_history_day{0}.csv'.format(day), usecols=[1], dtype=np.str).loc[:,'user'].values
    data_tmp.index = data_index
    data_dict = dict()
    total_size = data_index.shape[0]
    block_size = total_size//10
    print('day:{0}, total users:{1}'.format(day,total_size))
    start_time = time.time()
    i=0
    for u in data_tmp.index.value_counts().index:
        i += 1
        index = data_index[:] == u
        data_dict[u] = ss.csr_matrix(data_tmp.loc[index,:].sum().values)
        if (i+1)%block_size ==0:
            used_time = int(time.time()-start_time)
            print('day[{0}] in size[{1}%], total time:{2}'.format(day, int(i/total_size*100), used_time),)
    with open(data_path+'click_history_day{0}.pickl'.format(day), 'wb') as f:
        pickle.dump(data_dict, f)
    #with open(data_path+'click_history_day{0}.pickl'.format(day), 'rb') as f:
    #    test = pickle.load(f)
    #    print(test)
    
# ===========================================
# filelist: 
#         train:40428967,   count:9449445,
#     minitrain:4042898,   more_5:1544466
# miniminitrain:404291,   more_10:645381
#    test_click:4577464
# ===========================================

if __name__ == "__main__":
    
    # 解压提前压缩好的tar.gz文件, 
    # 主要用于解压上传到tinymind中的数据, 
    # 本地运行不需要解压 
    #ExtractData(FLAGS.data_dir)

    #设定参数
    #totalsize = {'minitrain':4042898, 'test_click':4577464}
    #file_size = totalsize[FLAGS.file_name]  #总的数据量
    #block_size = 100000  #数据块大小
    #C17s = GetC17IIndex(FLAGS.data_dir)
    param =[dict( 
        data_path = FLAGS.data_dir,
        file_name = FLAGS.file_name,
        chunksize = PARAMS.chunksize,
        output_path = FLAGS.output_dir,
        #C17s = C17s,
        day = d
                ) 
            for d in range(21,31)
            ]
    #ToPickleDump(param[5])
    #for p in param:
    #    ClickHistory(p)
    # 多进程处理onehot
    #OneHotEncoder(param[1])
    ClickHistory(param[-3])
    #exit()
    #MergeClickHistory666666(param[0])
    #with Pool(4) as p:  #4为进程数, 可改为更大
    #    pass
    #    p.map(ToPickleDump, param)
    with Pool(3) as p:  #4为进程数, 可改为更大
        p.map(ClickHistory, param)
    #with Pool(2) as p:  #4为进程数, 可改为更大
    #    p.map(MergeDayClickHistory, param)
    #param =[dict( 
    #data_path = FLAGS.data_dir,
    #file_name = FLAGS.file_name,)]
    #MergeAllClickHistory(param[0])

    
