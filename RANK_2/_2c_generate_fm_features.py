import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
import utils
from utils import *
import os

from sklearn.externals.joblib import dump, load

# 导入事先保存的从t0中取出部分列
t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')
# 取出事先保存的从t0中取出部分列
t0tv_mx = t0tv_mx_save['t0tv_mx']
# 取出 day==30 的 click
click_values = t0tv_mx_save['click']
# 取出 day
day_values = t0tv_mx_save['day']
print ("t0tv_mx loaded with shape", t0tv_mx.shape)

# 导入事先保存的 对t0 LabelEncoder 后的结果
t0 = load(utils.tmp_data_path + 't3a.joblib_dat')['t3a']
print ("t0 loaded with shape", t0.shape)

###################################################################################################

# 创建待处理特征, 并分类
vns={}
_vns1 = ['app_or_web', 'banner_pos', 'C1', 'C15', 'C16', 'C20',  'C14',
        'cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev',
        'cnt_device_ip_pday', 'dev_ip_cnt2', 'app_site_id', 'device_model']

vns['all_noid'] = _vns1
vns['all_withid'] = _vns1 + ['dev_id2plus']
vns['fm_5vars'] = ['app_or_web', 'banner_pos', 'C1', 'C15', 'device_model']
vns['all_but_ip'] = ['app_or_web', 'device_conn_type', 'C18', 'device_type',
       'banner_pos', 'C1', 'C15', 'C16', 'hour1', 'as_category', 'C21',
       'C19', 'C20', 'cnt_device_ip_day_hour', 'cnt_device_ip_pday',
       'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next',
       'dev_id_cnt2', 'dev_ip_cnt2', 'cnt_diff_device_ip_day_pday', 'C17',
       'C14', 'device_model', 'as_domain', 'app_site_id', '_A_app_or_web',
       '_A_C1', '_A_banner_pos', '_A_C16', '_A_C15', '_A_C18', '_A_C19',
       '_A_C21', '_A_C20', '_A_C17', '_A_C14', 'as_model', 'dev_id2plus']

# 生成fm的命令行参数
#cmd_str = utils.fm_path + ' -t 4 -s 8 -l 1e-5 /dev/shm/_tmp_2way_v.txt /dev/shm/_tmp_2way_t.txt'
# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
if utils.ffm: 
    is_ffm = 'FFM_'
    cmd_str = utils.ffm_path + 'ffm-train -t 4 -s 8 -l 1e-5 -p /dev/shm/_tmp_2way_v.txt  /dev/shm/_tmp_2way_t.txt /dev/shm/_tmp_2way_t.model'
    cmd_str_pdt = utils.ffm_path + 'ffm-predict /dev/shm/_tmp_2way_v.txt /dev/shm/_tmp_2way_t.model /dev/shm/_tmp_2way_v.txt.out'
else: 
    is_ffm = 'fm_'
    cmd_str = utils.fm_path + 'mf-train -t 6  -s 8  -f 5 -p /dev/shm/_tmp_2way_v.txt /dev/shm/_tmp_2way_t.txt /dev/shm/_tmp_2way_t.model'
    cmd_str_pdt = utils.fm_path + 'mf-predict -e 5 /dev/shm/_tmp_2way_v.txt /dev/shm/_tmp_2way_t.model /dev/shm/_tmp_2way_v.txt.out'
# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou

# 起始日期与终止日期
day_bgn = 22
day_end = 32

# 对某一类特征(vns.keys)中的所有的特征取值, 进行LabelEncoder
# 以前几天的LabelEncoder结果, 做为训练数据
# 以这一天的LabelEncoder结果, 做为校验数据
# 训练fm模型, 计算gini系数
# 并保存训练结果到fm_vecs 
fm_vecs = {}
for day_v in range(day_bgn, day_end):
    fm_vecs[day_v] = {}
    for vns_name in vns.keys():
        vns2 = vns[vns_name]
        # 生成准备处理的 日期以及 特征名
        print (day_v, vns_name)
        t1 = t0.loc[:, ['click']].copy()

        idx_base = 0
        # 对某一类特征(vns.keys)中的所有的特征取值, 进行LabelEncoder
        for vn in vns2:
            t1[vn] = t0[vn].values
            t1[vn] = np.asarray(t1[vn].astype('category').values.codes, np.int32) + idx_base
            idx_base = t1[vn].values.max() + 1  #用于LabelEncoder的累加
            #print ('-'* 5, vn, idx_base)
        
        # 生成待保存的文件名
        path1 = '/dev/shm/'
        fn_t = path1 + '_tmp_2way_t.txt'
        fn_v = path1 + '_tmp_2way_v.txt'
        
        print ("to write data files ...")
        #t1.loc[np.logical_and(day_values>=21, day_values < day_v),:].to_csv(open(fn_t, 'w'), sep='\t', header=False, index=False)
        #t1.loc[day_values==day_v,:].to_csv(open(fn_v, 'w'), sep='\t', header=False, index=False)
        #print (cmd_str)
        #fm_predv = pd.read_csv(open(path1 + '_tmp_2way_v.txt.out', 'r'), header=None).loc[:,0].values

        # changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
        if utils.ffm: 
            # 训练数据: 保存几天前的结果([21,day_v)) 
            generate_ffm_data_zhou(t1.loc[np.logical_and(day_values>=21, day_values < day_v),:], 'click', fn_t)
            # 校验数据: 保存当天的结果(day_v)
            generate_ffm_data_zhou(t1.loc[day_values==day_v,:], 'click', fn_v)
            # 进行一次训练
            os.system(cmd_str)
            os.system(cmd_str_pdt)  # changed by zhou
            # 读入结果
            print ("load results ...")
            fm_predv = pd.read_csv(open(path1 + '_tmp_2way_v.txt.out', 'r'), header=None).loc[:,0].values
        else:
            # 训练数据: 保存几天前的结果([21,day_v)) 
            generate_fm_data_zhou(t1.loc[np.logical_and(day_values>=21, day_values < day_v),:], 'click', fn_t)
            # 校验数据: 保存当天的结果(day_v)
            generate_fm_data_zhou(t1.loc[day_values==day_v,:], 'click', fn_v)
            # 进行一次训练
            os.system(cmd_str)
            os.system(cmd_str_pdt)
            # 读入结果
            print ("load results ...")
            fm_predv = get_fm_predict_zhou(fn_v, path1 + '_tmp_2way_v.txt.out')  # changed by zhou
        # changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou


        # 计算gini系数
        print ("--- gini_norm:", gini_norm(fm_predv, click_values[day_values==day_v], None))
        # 保存训练结果到fm_vecs
        fm_vecs[day_v][vns_name] = fm_predv
        print ('='*60)
        

####################################################################################################
# 取出fm_vecs中保存的FM训练结果 
# 保存到t2中, 未训练到(后几天)的结果转为0
# 并从t0中取出的43维特征值t0tv_mx, 和t2横向拼接在一起
# 最后按照t0tv_mx的格式保存拼接后结果t0tv_mx3 
# 其他t0tv_mx中值保持不变(如click,day)
t2 = t0.loc[:, ['click']].copy()
nn = t2.shape[0]
for vns_name in vns.keys():
    t2[vns_name] = np.zeros(nn)
    for day_v in range(day_bgn, day_end):
        print (day_v, vns_name)  
        # 取出fm_vecs中的训练结果, 重新保存到t2中
        t2.loc[day_values==day_v, vns_name] = fm_vecs[day_v][vns_name]
# 并保存t2到硬盘
#print ("to save features ...")
#dump(t2, tmp_data_path + 'FFM_t2.joblib_dat')
# changed by zhou
print ("to save {0} features ...".format(is_ffm))  
# changed by zhou
dump(t2, tmp_data_path + '{0}t2.joblib_dat'.format(is_ffm))

# 所从t0中取出的43维特征值, 同FM训练结果, 横向拼接在一起
t0tv_mx3 = np.concatenate([t0tv_mx[:, :43], t2.iloc[:, 1:].as_matrix()], axis=1)
#print ("t0tv_mx3 generated with shape", t0tv_mx3.shape)
# changed by zhou
print ("{0}t0tv_mx3 generated with shape".format(is_ffm), t0tv_mx3.shape)
t0tv_mx_save = {}
# 保存拼接后的结果
t0tv_mx_save['t0tv_mx'] = t0tv_mx3
# 保存 day==30 的 click
t0tv_mx_save['click'] = click_values
# 保留 日期
t0tv_mx_save['day'] = day_values
#dump(t0tv_mx_save, utils.tmp_data_path + 't0tv_mx3.joblib_dat')
# changed by zhou
dump(t0tv_mx_save, utils.tmp_data_path + '{0}t0tv_mx3.joblib_dat'.format(is_ffm))
