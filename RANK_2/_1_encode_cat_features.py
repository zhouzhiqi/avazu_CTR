import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
#sys.path.append('/home/zzhang/Downloads/xgboost/wrapper')
#import xgboost as xgb
from sklearn.externals.joblib import dump, load, Parallel, delayed
import utils
from utils import *

# 设定路径
raw_data_path = utils.raw_data_path
tmp_data_path = utils.tmp_data_path
train_name = utils.train_name
test_name = utils.test_name


# 读取文件
t0org0 = pd.read_csv(open(raw_data_path + train_name, "r+"), dtype = utils.dtypes)
h0org = pd.read_csv(open(raw_data_path + test_name, "r+"), dtype = utils.dtypes)

# 对训练数据随机采样
if utils.sample_pct < 1.0:
    # 随机数种子
    np.random.seed(999)
    # 生成随机数列, (0,1)之间, 共t0org0.shape[0]个
    r1 = np.random.uniform(0, 1, t0org0.shape[0])
    # 取随机数小于utils.sample_pct的数据为新的训练数据
    t0org0 = t0org0.loc[r1 < utils.sample_pct, :]
    print("testing with small sample of training data, ", t0org0.shape)

import gc
gc.collect()

# 为了与训练数据格式一致, 
# test添加'click'列, 并置为0
h0org['click'] = 0
# train和test拼接在一起
t0org = pd.concat([t0org0, h0org])
print ("finished loading raw data, ", t0org.shape)

###########################################################

print ("to add some basic features ...")
# 生成日期
t0org['day']=np.round(t0org.hour % 10000 / 100) 
# 生成小时
t0org['hour1'] = np.round(t0org.hour % 100)
# 从21日0点开始, 累计小时数
t0org['day_hour'] = (t0org.day.values - 21) * 24 + t0org.hour1.values
# 前一小时
t0org['day_hour_prev'] = t0org['day_hour'] - 1
# 后一小时
t0org['day_hour_next'] = t0org['day_hour'] + 1
# app_id=='ecad2386'
t0org['app_or_web'] = 0
# app_id=='ecad2386'
t0org.loc[t0org.app_id.values=='ecad2386', 'app_or_web'] = 1

#############################################################################################################


# 重新命名待处理文件
t0 = t0org

# app和site两个信息相似, 
# 有一定的互补性, 直接相加
t0['app_site_id'] = np.add(t0.app_id.values, t0.site_id.values)

print( "to encode categorical features using mean responses from earlier days -- univariate")
#sys.stdout.flush()

# 在t0中添加'app_or_web'的 前几天的点击概率
calc_exptv(t0,  ['app_or_web'])
gc.collect() # changed by zhou
exptv_vn_list = ['app_site_id', 'as_domain', 'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                'app_site_model', 'site_model','app_model', 'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']
# 在t0中添加exptv_vn_list中的特征的 前几天的点击概率, 
# 如果t0中没有exptv_vn_list的特征, 就创建新特征
calc_exptv(t0, exptv_vn_list)
gc.collect() # changed by zhou
# 在t0中添加'app_site_id'的 前几天的点击概率 以及浏览次数
calc_exptv(t0, ['app_site_id'], add_count=True)
gc.collect() # changed by zhou
#########################################################################################################

print ("to encode categorical features using mean responses from earlier days -- multivariate")
vns = ['app_or_web',  'device_ip', 'app_site_id', 'device_model', 'app_site_model', 'C1', 'C14', 'C17', 'C21',
                        'device_type', 'device_conn_type','app_site_model_aw', 'dev_ip_app_site']
dftv = t0.loc[np.logical_and(t0.day.values >= 21, t0.day.values < 32), ['click', 'day', 'id'] + vns].copy()
# 创建新特征, to 'str' 直接相加
dftv['app_site_model'] = np.add(dftv.device_model.values, dftv.app_site_id.values)
dftv['app_site_model_aw'] = np.add(dftv.app_site_model.values, dftv.app_or_web.astype('str').values)
dftv['dev_ip_app_site'] = np.add(dftv.device_ip.values, dftv.app_site_id.values)
# 把vns中的特征 转为类别型变量
for vn in vns:
    dftv[vn] = dftv[vn].astype('category')
    print( vn)
    
# 为特征添加下限, (点击次数+n_k*总平均点击次数) / (浏览次数+n_k)
n_ks = {'app_or_web': 100, 'app_site_id': 100, 'device_ip': 10, 'C14': 50, 'app_site_model': 50, 'device_model': 100, 'device_id': 50,
        'C17': 100, 'C21': 100, 'C1': 100, 'device_type': 100, 'device_conn_type': 100, 'banner_pos': 100,
        'app_site_model_aw': 100, 'dev_ip_app_site': 10 , 'device_model': 500}

exp2_dict = {}
for vn in vns:  #生成0向量
    exp2_dict[vn] = np.zeros(dftv.shape[0])

days_npa = dftv.day.values
    
for day_v in range(22, 32):
    gc.collect() # changed by zhou
    # 取前几天的数据 做训练集
    df1 = dftv.loc[np.logical_and(dftv.day.values < day_v, dftv.day.values < 31), :].copy()
    # 取当天的数据 做测试集
    df2 = dftv.loc[dftv.day.values == day_v, :]
    print ("Validation day:", day_v, ", train data shape:", df1.shape, ", validation data shape:", df2.shape)
    # 生成前几天的一样的平均点击率
    pred_prev = df1.click.values.mean() * np.ones(df1.shape[0])
    # 删除df1中与'exp2_'有关的列
    for vn in vns:
        if 'exp2_'+vn in df1.columns:
            df1.drop('exp2_'+vn, inplace=True, axis=1)

    # 处理df1
    for i in range(3): #训练4次, 每次开0.25次方(np.power)
        for vn in vns:
            # 统计前几天的点击概率
            p1 = calcLeaveOneOut2(df1, vn, 'click', n_ks[vn], 0, 0.25, mean0=pred_prev)
            # 前几天平均点击率 * 分组后前几天的点击概率
            pred = pred_prev * p1
            print (day_v, i, vn, "change = ", ((pred - pred_prev)**2).mean())
            pred_prev = pred    
        
        # 前几天的平均点击概率
        pred1 = df1.click.values.mean()
        for vn in vns:
            print( "="*20, "merge", day_v, vn)
            # 统计当天的点击概率
            diff1 = mergeLeaveOneOut2(df1, df2, vn)
            # 前几天的平均点击概率 * 分组后当天的点击概率
            pred1 *= diff1
            # 重新赋值当天的点击概率
            exp2_dict[vn][days_npa == day_v] = diff1
        
        pred1 *= df1.click.values.mean() / pred1.mean()
        print( "logloss = ", logloss(pred1, df2.click.values))
        #print( my_lift(pred1, None, df2.click.values, None, 20, fig_size=(10, 5)))
        #plt.show()

# 把生成好的'exp2_', 添加到t0中
for vn in vns:
    t0['exp2_'+vn] = exp2_dict[vn]

############################################################################################

print( "to count prev/current/next hour by ip ...")
# 对device_ip在时间维度上进行处理, 把当前小时的 不同device_ip出现的次数, 扩展到其它小时
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour_prev', fill_na=0)
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour', fill_na=0)
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour_next', fill_na=0)

print( "to create day diffs")
t0['pday'] = t0.day - 1
# 对device_ip在日期维度上进行处理, 把当前日期的 不同device_ip出现的点击概率和浏览次数, 扩展到前一天
calcDualKey(t0, 'device_ip', None, 'day', 'pday', 'click', 10, None, True, True)
# 把当前日期的 不同device_ip出现的浏览次数, 减去前一天的浏览次数
t0['cnt_diff_device_ip_day_pday'] = t0.cnt_device_ip_day.values  - t0.cnt_device_ip_pday.values
# 当前网络时间
t0['hour1_web'] = t0.hour1.values
# 把非app_or_web的时间 转为-1
t0.loc[t0.app_or_web.values==0, 'hour1_web'] = -1
# 不同device_ip含有的 不同app_id的个数
t0['app_cnt_by_dev_ip'] = my_grp_cnt(t0.device_ip.values.astype('str'), t0.app_id.values.astype('str'))

# 生成小时
t0['hour1'] = np.round(t0.hour.values % 100)
# 把当前日期的 不同device_ip出现的浏览次数, 减去前一天的浏览次数
t0['cnt_diff_device_ip_day_pday'] = t0.cnt_device_ip_day.values  - t0.cnt_device_ip_pday.values

# 不同device_ip含有的 不同id的进行LabelEncoder
t0['rank_dev_ip'] = my_grp_idx(t0.device_ip.values.astype('str'), t0.id.values.astype('str'))
# 不同device_ip+day含有的 不同id的进行LabelEncoder
t0['rank_day_dev_ip'] = my_grp_idx(np.add(t0.device_ip.values, t0.day.astype('str').values).astype('str'), t0.id.values.astype('str'))
# 不同device_ip+app_id含有的 不同id的进行LabelEncoder
t0['rank_app_dev_ip'] = my_grp_idx(np.add(t0.device_ip.values, t0.app_id.values).astype('str'), t0.id.values.astype('str'))

# 不同device_ip 出现的次数(含有的不同id的个数)
t0['cnt_dev_ip'] = get_agg(t0.device_ip.values, t0.id, np.size)
# 不同device_id 出现的次数(含有的不同id的个数)
t0['cnt_dev_id'] = get_agg(t0.device_id.values, t0.id, np.size)
# 对不同device_id 出现的总次数进行截断, 最大出现次数为300
t0['dev_id_cnt2'] = np.minimum(t0.cnt_dev_id.astype('int32').values, 300)
# 对不同device_ip 出现的总次数进行截断, 最大出现次数为300
t0['dev_ip_cnt2'] = np.minimum(t0.cnt_dev_ip.astype('int32').values, 300)

gc.collect() # changed by zhou
# 对只出现过一次的device_id, 重新命名为'___only1'
t0['dev_id2plus'] = t0.device_id.values
t0.loc[t0.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
# 对只出现过一次的device_ip, 重新命名为'___only1'
t0['dev_ip2plus'] = t0.device_ip.values
t0.loc[t0.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'
# (t0.app_or_web * 2 - 1) == ((0,1) -> (-1,1))
# 当前小时, 不同device_ip 出现的次数 减去 前一小时, 对应device_ip 出现的次数
t0['diff_cnt_dev_ip_hour_phour_aw2_prev'] = (t0.cnt_device_ip_day_hour.values - t0.cnt_device_ip_day_hour_prev.values) * ((t0.app_or_web * 2 - 1)) 
# 当前小时, 不同device_ip 出现的次数 减去 后一小时, 对应device_ip 出现的次数
t0['diff_cnt_dev_ip_hour_phour_aw2_next'] = (t0.cnt_device_ip_day_hour.values - t0.cnt_device_ip_day_hour_next.values) * ((t0.app_or_web * 2 - 1)) 

################################################################################################

print( "to save t0 ...")
print( t0.shape )
# 保存处理好的t0
dump(t0, tmp_data_path + 't0.joblib_dat')
to_tar(tmp_data_path, 't0.joblib_dat')

###################################################################################################

# 从t0中取出部分列保存
print ("to generate t0tv_mx .. ")
app_or_web = None
# 起始日期
_start_day = 22
# 原始特征
list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list_dict = {}
# 后加特征
feature_list_name = 'tvexp3' 
feature_list_dict[feature_list_name] = list_param + \
                            ['exptv_' + vn for vn in ['app_site_id', 'as_domain', 
                             'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                             'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']] + \
                            ['cnt_diff_device_ip_day_pday', 
                             'app_cnt_by_dev_ip', 'cnt_device_ip_day_hour', 'app_or_web',
                             'rank_dev_ip', 'rank_day_dev_ip', 'rank_app_dev_ip',
                             'diff_cnt_dev_ip_hour_phour_aw2_prev', 'diff_cnt_dev_ip_hour_phour_aw2_next',
                             'exp2_device_ip', 'exp2_app_site_id', 'exp2_device_model', 'exp2_app_site_model',
                             'exp2_app_site_model_aw', 'exp2_dev_ip_app_site',
                             'cnt_dev_ip', 'cnt_dev_id', 'hour1_web']
# 22 <= -- <= 30
filter_tv = np.logical_and(t0.day.values >= _start_day, t0.day.values < 31)
# 22 <= -- <= 29
filter_t1 = np.logical_and(t0.day.values < 30, filter_tv)
# 29 <  -- <= 30 ==30
filter_v1 = np.logical_and(~filter_t1, filter_tv)    
    
print( filter_tv.sum())

# 输出缺失的特征名(上面程序没有处理)
for vn in feature_list_dict[feature_list_name] :
    if vn not in t0.columns:
        print( "="*60 + vn)
        
# 验证集的click
yv = t0.click.values[filter_v1]
# 只保存feature_list_name中的列, 并转为ndarray
t0tv_mx = t0.as_matrix(feature_list_dict[feature_list_name])

print( t0tv_mx.shape)


print( "to save t0tv_mx ...")
# 保存从t0中取出的43列特征, click, day以及site_id
t0tv_mx_save = {}
t0tv_mx_save['t0tv_mx'] = t0tv_mx
t0tv_mx_save['click'] = t0.click.values
t0tv_mx_save['day'] = t0.day.values
t0tv_mx_save['site_id'] = t0.site_id.values
dump(t0tv_mx_save, tmp_data_path + 't0tv_mx.joblib_dat')
to_tar(tmp_data_path, 't0tv_mx.joblib_dat')
###############################################################################


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
to_tar(tmp_data_path, 't3a.joblib_dat')