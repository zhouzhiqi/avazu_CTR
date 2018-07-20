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

#####################################################################################################

# 事先生成的t0
t0 = load(utils.tmp_data_path + 't0.joblib_dat')
print ("t0 loaded with shape", t0.shape)

# 对不同device_id 出现的总次数进行截断, 最大出现次数为300
t0['dev_id_cnt2'] = np.minimum(t0.cnt_dev_id.astype('int32').values, 300)
# 对不同device_ip 出现的总次数进行截断, 最大出现次数为300
t0['dev_ip_cnt2'] = np.minimum(t0.cnt_dev_ip.astype('int32').values, 300)

# 对只出现过一次的device_id, 重新命名为'___only1'
t0['dev_id2plus'] = t0.device_id.values
t0.loc[t0.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
# 对只出现过一次的device_ip, 重新命名为'___only1'
t0['dev_ip2plus'] = t0.device_ip.values
t0.loc[t0.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'

# 当前小时, 不同device_ip 出现的次数 是否等于 前一天, 对应device_ip 出现的次数
t0['device_ip_only_hour_for_day'] = t0.cnt_device_ip_day_hour.values == t0.cnt_device_ip_pday.values

######################################################################################################

# 生成新的组合特征 app_site_id 与 vns0 的组合, 以str相加
vns0 = ['app_or_web', 'banner_pos', 'C1', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
for vn in vns0 + ['C14']:
    print (vn)
    vn2 = '_A_' + vn  #新的特征名
    # 以str相加
    t0[vn2] = np.add(t0['app_site_id'].values, t0[vn].astype('str').values)
    t0[vn2] = t0[vn2].astype('category')

##############################################################################################################
# 对t0全部的特征取值, 进行LabelEncoder
t3 = t0
# 事先生成待处理的特征名
vns1 = vns0 + ['hour1'] + ['_A_' + vn for vn in vns0] + \
 ['device_model', 'device_type', 'device_conn_type', 'app_site_id', 'as_domain', 'as_category',
      'cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
     'cnt_diff_device_ip_day_pday', 'as_model'] + \
    [ 'dev_id_cnt2', 'dev_ip_cnt2', 'C14', '_A_C14', 'dev_ip2plus', 'dev_id2plus']
#'cnt_device_ip_day', 'device_ip_only_hour_for_day'
    
t3a = t3.loc[:, ['click']].copy()
idx_base = 3000
for vn in vns1:
    # 处理浏览次数类特征, 并进行截断处理, 范围[-100, 200], 再LabelEncoder
    if vn in ['cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
     'cnt_diff_device_ip_day_pday', 'cnt_device_ip_day', 'cnt_device_ip_pday']:
        _cat = pd.Series(np.maximum(-100, np.minimum(200, t3[vn].values))).astype('category').values.codes
    # 合并domain: app_domain, site_domain 以str相加, 再LabelEncoder
    elif vn in ['as_domain']:
        _cat = pd.Series(np.add(t3['app_domain'].values, t3['site_domain'].values)).astype('category').values.codes
    # 合并category: app_category, site_category 以str相加, 再LabelEncoder
    elif vn in ['as_category']:
        _cat = pd.Series(np.add(t3['app_category'].values, t3['site_category'].values)).astype('category').values.codes
    # 合并model: app_site_id, device_model 以str相加, 再LabelEncoder
    elif vn in ['as_model']:
        _cat = pd.Series(np.add(t3['app_site_id'].values, t3['device_model'].values)).astype('category').values.codes
    # 其他特征: 直接LabelEncoder
    else:
        _cat = t3[vn].astype('category').values.codes
    # 更改数据类型为int32, 并转为ndarray
    _cat = np.asarray(_cat, dtype='int32')
    # LabelEncoder后的结果 加上idx_base偏置
    _cat1 = _cat + idx_base
    # 把亲的数据添加入t3a
    t3a[vn] = _cat1
    print (vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
    # 更新idx_base偏置
    idx_base += _cat.max() + 1

##################################################
# idx_base += _cat.max() + 1
# 对t0全部的特征取值, 进行LabelEncoder, 所以要不断叠加, 
# 加入idx_base偏置是保证LabelEncoder
##################################################

# 保存t3a 以及对应的idx_base偏置
print ("to save t3a ...")
t3a_save = {}
t3a_save['t3a'] = t3a
t3a_save['idx_base'] = idx_base  
dump(t3a_save, utils.tmp_data_path + 't3a.joblib_dat')
