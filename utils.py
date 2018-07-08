# coding: utf-8

import numpy as np
import pandas as pd
import time
import gc
import os
import scipy.sparse as ss
import xgboost as xgb
import tarfile
from multiprocessing import Pool
import pickle as pickle

"""--------------------分割线--------------------"""
# 加载数据
class DataUtils(object):
    """关于数据加载的小工具"""

    def __init__(self,):
        self.total_size = 0  #已经读取的总数据量
        self.data_over = False  #数据是否读取完全

    def ExtractData(self, data_path):
        """把data_path中的tar.gz文件全部解压出来"""
        file_list = os.listdir(data_path)  #文件列表
        if 'count.csv' in file_list: return None  #若已经解压, pass
        for file_name in file_list: #把所有.tar.gz解压到data_path
            filename, extension = os.path.splitext(file_name)  #分解文件名和后缀名
            if extension=='.gz':  #如果后缀名为'.gz', 则解压到data_path
                # 打开'*.tar.gz'文件, 
                with tarfile.open(data_path + file_name, 'r:gz') as tar: 
                    tar.extractall(data_path)  #解压到data_path

    def LoadData(self, data_path, file_name, use_cols=None):
        """
        迭代加载数据
        
        data_path: 文件路径,
        file_name: 文件名称,不用加'.csv'
        use_cols: 专门读取某几列,
        return: 可用于.get_chunk()的数据
        """
        self.total_size = 0  #已经读取的数据量
        self.data_over = False  #数据是否读取完全
        data = pd.read_csv(data_path+'{0}.csv'.format(file_name),  
                            usecols=use_cols,
                            iterator=True)  #用pandas迭代读取
        print('load data: ', data_path+'{0}.csv'.format(file_name))
        return data


    def JumpData(self, data, data_begin, chunksize, ):
        """
        跳过前 data_begin 条数据
        
        data: 可用.get_chunk()迭代读取
        data_begin: 跳过的总数据量
        chunksize: 每次迭代跳过的数据量
        return: 跳过后的可迭代数据
        """
        #print('jump data')
        if data_begin <= 0: return data  #如果跳过0个, 不跳, 直接返回
        while True:
            data_tmp = self.NextChunk(data, chunksize)  #每次读取chunk_size条数据 
            # 如果总的数据量小于 开始时的数据量, pass
            if self.total_size < data_begin: continue  
            # 如果总的数据量大于 开始时的数据量, 往下进行
            else: break  
            gc.collect()
        # 输出跳过数据量
        print('{0} data has jumped'.format(self.total_size))
        return data

    def NextChunk(self, data, chunksize):
        """
        获取 chunksize 条数据
        
        data: 可用.get_chunk()迭代读取
        chunksize: 每次迭代跳过的数据量
        """
        try: data_tmp = data.get_chunk(chunksize)  #每次读取chunk_size条数据 
        except StopIteration: self.data_over = True  #读取结束后跳出循环
        else: 
            self.total_size += data_tmp.shape[0]  #累加当前数据量
            return data_tmp  #返回取出的数据


"""--------------------分割线--------------------"""
# Onehot编码
class OnehotUtils(object):
    """独执编码小工具"""

    def __init__(self,):
        pass

    def GetOneHot_Cat(self, data_tmp, get_index, ):
        """
        对'类别变量'进行OneHot编码,
        
        data_tmp: 数据
        get_index: 通过组合好的特征名, 获取索引
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        X_train_cat = self.CategoryEncoder(data_tmp, get_index)
        #print(X_train_cat.shape)

        target = data_tmp.loc[:, 'click'].values
        y_train = ss.csc_matrix(target)  #转成 列 稀疏矩阵
        #print(y_train.shape)

        return X_train_cat, y_train

    def GetCategoryColums(self, data_path, file_name='count', threshold=10):
        """
        获取OneHot编码后的列名columns
        
        data_path: 已经计数好的文件所在路径
        file_name: 已经计数好的文件的名子
        threshold: 取频率>=threshold的特征值Onehot
        return: 返回编码后的onehot列名columns
        """
        print('get count ')
        # 读取计数文件, 第一列为index (index_col=0)
        count = pd.read_csv(data_path+'{0}.csv'.format(file_name), index_col=0) 
        # 找出频率>=threshold 的特征名与特征值的组合, 做为OneHot的列名columns,
        more_th = count[count['total']>=threshold].index  
        # 把频率<threshold的组合特征统一置为'less_threshold'
        more_th = more_th.append(pd.Index(['less_threshold']))  
        # 把OneHot的列名columns, 改造成: 特征名与特征值的组合 做为 下标索引, 顺序id 做为 取值
        get_index = pd.Series(data=np.arange(more_th.shape[0]), 
                                index=more_th, dtype=np.int64)
        #简言之: 名称 索引 -> ID 索引
        print('the number of columns is:', len(get_index))
        del count, more_th
        gc.collect()
        return get_index

    def CategoryEncoder(self, data_tmp,  get_index):
        """
        对类别型变量进行onehot编码,
        
        data_tmp: 数据
        get_index: 通过组合好的特征名, 获取索引
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        # 初始化onehot数组, 全部为0, 有值置1     
        # get_index的[-1],是用来累加那些频率小于10的特征取值
        onehot = np.zeros((data_tmp.shape[0],get_index.shape[0]), dtype=np.int8)
        # 把特征名称与特征取值结合起来, 做为get_index的索引
        for c in data_tmp.columns: 
            if c in ['id', 'click', 'hour']: continue #跳过非类别型变量
            data_tmp.loc[:, c] = data_tmp.loc[:, c].map(lambda x: c+'='+str(x))
        # OneHot编码, 对符合条件的位置, 赋值 1
        for i in np.arange(data_tmp.shape[0]):
            index = data_tmp.iloc[i,3:]  #取出['C1':]的values(合成过)
            for c in index:  #逐个values去get_index的索引
                try: j =  get_index[c] #找到索引, 赋值 1
                except KeyError:  #找不到索引, 说明频率小于10,
                    onehot[i, -1] += 1  #在末尾累加
                    continue   
                onehot[i, j] = 1  #对应位置赋值为1
        
        return ss.csr_matrix(onehot)  #转成 行 稀疏矩阵

##################################################

    def GetOneHot_Hour(self, data_tmp, get_index ):
        """
        对'时间变量'进行OneHot编码,
        
        data_tmp: 数据
        get_index: 没用,保持参数数量一致
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        X_train_hour = self.SplitHour(data_tmp)
        #print(X_train_hour.shape)

        target = data_tmp.loc[:, 'click'].values
        y_train = ss.csc_matrix(target)  #转成 列 稀疏矩阵
        #print(y_train.shape)

        return X_train_hour, y_train

    def SplitHour(self, data_tmp):
        """
        对时间进行拆分, 并onehot
        
        data_tmp: 数据
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        # 拆分后的新特征 ['workingday':1-5,6-7, 'week'1~7,  'hour':0~23,
        # 'certaintimes':0-4,5-9,10-14,15-19,20-23, ] 共38列
        split_hour = np.zeros((data_tmp.shape[0], 38), dtype=np.int8)
        # 拆分出小时
        hour = data_tmp.loc[:, 'hour'].values %14100000 %100
        # 拆分出星期
        week = ((data_tmp.loc[:, 'hour'].values %14100000 //100) %7 +2) %7
        #split_hour[:,-1] = hour  #不onehot
        #split_hour[:,-2] = week  #不onehot
        # OneHot编码, 对符合条件的位置, 赋值 1
        for i in np.arange(data_tmp.shape[0]):
            split_hour[i, week[i]//5] = 1  #0-1
            split_hour[i, week[i]+1] = 1   #2-8
            split_hour[i, hour[i]+9] = 1   #9-32
            split_hour[i, hour[i]//5+33] = 1  #33-37

        return ss.csr_matrix(split_hour)

##################################################

    def GetOneHot_XGB(self, data_path, file_name, model_path, 
                        data_begin, num_trees, deep, onehot_name ):
        """
        对'xgboost的输出'进行OneHot编码,
        
        data_tmp: 数据
        model_path: xgboost的存储路径
        num_trees: xgboost的树的数量
        deep: xgboost的树的深度
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        print('load data: ', data_begin)
        X_train, y_train = NpzUtils().LoadNpz(data_path, file_name, data_begin, onehot_name,)
        y_train_tmp = y_train.toarray().astype(np.float32)[0]

        X_train_xgb = self.XgboostEncoder(X_train, y_train_tmp, 
                                model_path, num_trees, deep, onehot_name )
        #print(X_train_xgb.shape)

        return X_train_xgb, y_train

    def XgboostEncoder(self, X_train, y_train, model_path,  num_trees, deep, onehot_name ):
        """
        对xgboost输出的结点位置文件, 进行onehot
        
        model_path: xgboost的存储路径
        num_trees: xgboost的树的数量
        deep: xgboost的树的深度
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        # 展开后的维数:每颗树实际有2**(deep+1)个结点, deep为模型的参数max_depth
        length = 2**(deep+1)
        # 生成空白onehot矩阵, 用于赋值为1,  
        leaf_index = np.zeros((X_train.shape[0], num_trees*length), dtype=np.int8)
        #转为xgb专用数据格式
        #print('to xgb.DMatrix')
        xgtrain = xgb.DMatrix(X_train, label = y_train,)
        # 导入模型
        xgb_model = xgb.Booster(model_file=model_path + 
                                'tree{0}_deep{1}_{2}.xgboost'.format(num_trees, deep, onehot_name))
        # 开始预测
        print('xgboost predict . . .')
        new_feature = xgb_model.predict(xgtrain, pred_leaf=True)  
        # pred_leaf=True, 输出叶子结点索引
        # 对新特征onehot编码
        for i in np.arange(X_train.shape[0]):
            for tree in np.arange(num_trees):  
                # tree*length是每颗树索引的区域块, 
                # new_feature[i,tree]是该颗树的叶子结点索引
                j = tree*length + new_feature[i,tree]
                leaf_index[i, j] = 1

        return ss.csr_matrix(leaf_index)

##################################################

    def GetOneHot_His(self, data_tmp, day_clicks, ):
        """
        对数据进行OneHot编码,
        
        data_tmp: 数据
        day_clicks: 每天,每个user的点击历史
        return: 返回稀疏矩阵scipy.sparse, 不包含'id'
        """
        # 把事先生成的每天,每个user的点击历史合并起来(day_clicks)
        # 遍历所有user, 生成前三天的点击历史
        hour_day_user_C17 = self.HourDayUserC17(data_tmp)
        X_train = self.ClickHistory(hour_day_user_C17, day_clicks)
        #print(X_train.shape)

        target = data_tmp.loc[:, 'click'].values
        y_train = ss.csc_matrix(target)  #转成 列 稀疏矩阵
        #print(y_train.shape)

        return X_train, y_train

    def MergeAllClickHistory(self, data_path, ):
        """
        合并已经生成好的 按天存储的 每个user的点击历史(.csv)
        
        data_path: 生成好的文件的存储位置
        return: 合并好的每天,每个user的点击历史
        return使用方法: click_all[day].loc['user',:]
        """
        # 日期
        days = [d for d in range(21,31)]
        # 生成字典,日期为key
        click_all = dict()
        start_time = time.time()
        print('merge clicks')
        for d in days:  #按日期为key生成
            # 读取点击历史
            click_history = pd.read_csv(data_path+'click_history_day{0}.csv'.format(d), 
                                        usecols=[2,3,4,5], dtype=np.int32)
            # 读取user
            user = pd.read_csv(data_path+'click_history_day{0}.csv'.format(d), 
                                usecols=[1],)
            # 把user置为index
            click_history.index = user.iloc[:,0].values
            # 按日期为key生成: click_all[day].loc['user',:]
            click_all[d] = click_history
        print('merge all click cost time:', int(time.time() -  start_time))

        return click_all

    def HourDayUserC17(self, data_tmp,):
        """
        把传入的数据处理为['hour', 'day', 'user', 'C17']
        
        data_tmp: 数据
        return: 处理好的数据
        """
        # 取出相关列
        data_tmp = data_tmp.iloc[:, [1,2,3,4,11,12,13,19]]
        # 对相关列重新命名
        data_tmp.columns = ['click', 'hour', 'day', 'user', 
                            'device_id', 'device_ip', 'device_model','C17']
        # 生成日期
        data_tmp.loc[:,'day'] = data_tmp.loc[:, 'hour'].values %14100000 //100
        # 生成小时
        data_tmp.loc[:,'hour'] = data_tmp.loc[:, 'hour'].values %14100000 %100
        # 生成user
        data_tmp.loc[:,'user'] = data_tmp.loc[:, 'device_id'].values
        index = data_tmp.loc[:, 'device_id']=='a99f214a'
        data_tmp.loc[index, 'user'] = data_tmp.loc[index, 'device_ip']  + '&' + \
                                        data_tmp.loc[index, 'device_model']
        
        return data_tmp.loc[:, ['hour', 'day', 'user', 'C17']]


    def GetOneDay(self, day_clicks,  hour, days, user, C17):
        """
        生成某一天的点击历史
        
        day_clicks: 每天,每个user的点击历史
        hour: 小时
        days: 日期
        user: 用户
        C17: 正在浏览广告类型
        return: onehot后的数据
        """
        click_tmp = np.zeros((10),dtype=np.int8)
        # 传入列名
        # ['click_times', 'total_times', 'C17_Fir', 'C17_Sec']
        # onehot后的列名
        # ['click_times<=1', '1<click_times<=5', '5<click_times<=10', '10<click_times', 
        #  'total_times<=5', '5<total_times<=20', '20<total_times<=100', '100<total_times', 
        #  'is_C17_Fir', 'is_C17_Sec']
        # 这一day, 这个user, 是否有点击历史, 没有返回0, 有进行处理
        try: click_hour = day_clicks[days].loc[user,:].values
        except KeyError: return click_tmp
        # 处理点击次数
        click_times = click_hour[0]
        if     click_times <= 1: click_tmp[0]=1
        if 1 < click_times <= 5: click_tmp[1]=1
        if 5 < click_times <= 10: click_tmp[2]=1
        if 10 < click_times: click_tmp[3]=1
        # 处理浏览次数
        total_times = click_hour[1]
        if     total_times <= 5: click_tmp[4]=1
        if 5 < total_times <= 20: click_tmp[5]=1
        if 20 < total_times <= 100: click_tmp[6]=1
        if 100 < total_times: click_tmp[7]=1
        # 处理感兴趣广告类型
        if click_hour[-1] == C17: click_tmp[-1]=1
        if click_hour[-2] == C17: click_tmp[-2]=1
        
        return click_tmp

    def GetThreeDay(self, day_clicks, H_D_U_C17 ):
        """
        生成某user前三天 的点击历史

        day_clicks: 每天,每个user的点击历史
        H_D_U_C17: 列名为['hour', 'day', 'user', 'C17']
        return:  返回稀疏矩阵scipy.sparse
        """
        # 生成'hour', 'day', 'user', 'C17'
        hour = H_D_U_C17['hour']
        days = H_D_U_C17['day']
        user = H_D_U_C17['user']
        C17 = H_D_U_C17['C17']
        # 生成前一天的点击历史
        click_tmp = self.GetOneDay(day_clicks, hour, days-1, user, C17)
        # 生成前二天的点击历史
        click_tmp = np.concatenate((click_tmp, 
                    self.GetOneDay(day_clicks, hour, days-2, user, C17)))
        # 生成前三天的点击历史
        click_tmp = np.concatenate((click_tmp, 
                    self.GetOneDay(day_clicks, hour, days-3, user, C17)))

        return ss.csr_matrix(click_tmp)

    def ClickHistory(self, hour_day_user_C17, day_clicks):
        """
        对每个user的前三天的点击历史进行onehot
        
        hour_day_user_C17: 经过处理的所有的user
        day_clicks: 每天,每个user的点击历史
        return: 所有user的前三天的点击历史
        """
        # 生成第一个user的 前三天的点击历史
        H_D_U_C17 = hour_day_user_C17.iloc[0,:]
        click_history = self.GetThreeDay(day_clicks, H_D_U_C17 )
        # 循环生成第i个user的 前三天的点击历史, 并不断拼接在一起
        for i in np.arange(1, hour_day_user_C17.shape[0]):
            # 生成第i个user处理后的数据
            H_D_U_C17 = hour_day_user_C17.iloc[i,:]
            # 生成第i个user的 前三天的点击历史
            click_tmp = self.GetThreeDay(day_clicks, H_D_U_C17)
            # 不同user的点击历史, 并不断拼接在一起
            click_history = ss.vstack((click_history, click_tmp))

        return click_history

##################################################
# ss.npz
class NpzUtils(object):
    """ss.npz小工具"""
    
    def SaveNpz(self, X_train, y_train, output_path, file_name, data_begin, onehot_name,):
        """保存稀疏矩阵"""
        assert onehot_name in ['A_cat', 'A_hour', 'A_his', 'B_cat', 'B_cat_xgb',
                               'A_cat_xgb','A_hour_xgb','A_his_xgb',], "onehot name error"
        print(X_train.shape, ' , ', y_train.shape)  #输出文件shape

        if data_begin != 'all': data_begin = 'begin{0}'.format(data_begin)

        ss.save_npz(output_path+'{0}_X_{1}_{2}'.format(
                            file_name, onehot_name, data_begin), X_train)
        ss.save_npz(output_path+'{0}_y_{1}_{2}'.format(
                            file_name, onehot_name, data_begin), y_train)
        print('saved in: ', output_path)
        
    def LoadNpz(self, data_path, file_name, data_begin, onehot_name,):
        """导入稀疏矩阵"""
        assert onehot_name in ['A_cat', 'A_hour', 'A_his', 'B_cat', 'B_cat_xgb',
                               'A_cat_xgb','A_hour_xgb','A_his_xgb',], "onehot name error"
        if data_begin != 'all': data_begin = 'begin{0}'.format(data_begin)

        X_train = ss.load_npz(data_path+'{0}_X_{1}_{2}.npz'.format(
                            file_name, onehot_name, data_begin,))
        y_train = ss.load_npz(data_path+'{0}_y_{1}_{2}.npz'.format(
                            file_name, onehot_name, data_begin,))

        return X_train, y_train


    def MergeNpz(self, output_path, file_name, data_begins, onehot_name,):
        """把生成的多个.npz合并成一个文件并保存"""
        #导入从0开始的文件, 为下面的合并做准备
        assert onehot_name in ['A_cat', 'A_hour', 'A_his', 'B_cat', 'B_cat_xgb',
                               'A_cat_xgb','A_hour_xgb','A_his_xgb',], "onehot name error"

        X_train, y_train = self.LoadNpz(output_path, 
                            file_name, data_begins[0], onehot_name,)

        for begin in range(1,len(data_begins)): #循环读入文件
            #文件暂存
            X_train_tmp, y_train_tmp = self.LoadNpz(output_path, 
                                    file_name, data_begins[begin], onehot_name,)
            X_train = ss.vstack((X_train, X_train_tmp))  #与原有 行稀疏矩阵 进行 行连接
            y_train = ss.hstack((y_train, y_train_tmp))  #与原有 列稀疏矩阵 进行 列连接
        
        #y_train.toarray().astype(np.float32)[0]
        print('total shape: ', X_train.shape, ' , ', y_train.shape)
        #保存最终连接完成的稀疏矩阵
        ss.save_npz(output_path+'{0}_X_{1}_{2}'.format(file_name, onehot_name, 'all'), X_train)
        ss.save_npz(output_path+'{0}_y_{1}_{2}'.format(file_name, onehot_name, 'all'), y_train)
        print('saved in {0}'.format(output_path))

        #return X_train, y_train





















