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
# 导入保存好的从t0中取出的43维特征值
t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')
# 取出对应数据
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
site_id_values= t0tv_mx_save['site_id']
print ("t0tv_mx loaded")

# changed by zhou changed by zhou changed by zhou 
if utils.ffm: 
    is_ffm = 'FFM_'
else: 
    is_ffm = 'fm_'
# changed by zhou changed by zhou changed by zhou

# 是否以31号为校验集
day_test = 30
if utils.tvh == 'Y':
    day_test = 31

#RandomForest model output
rf_pred = load(utils.tmp_data_path + '{0}rf_pred_v.joblib_dat'.format(is_ffm))
print ("RF prediction loaded with shape", rf_pred.shape)

#GBDT (xgboost) model output
xgb_pred = load(utils.tmp_data_path + '{0}xgb_pred_v.joblib_dat'.format(is_ffm))
print ("xgb prediction loaded with shape", xgb_pred.shape)

#Vowpal Wabbit model output
ctr = 0
vw_pred = 0
for i in [1, 2, 3, 4]:
    vw_pred += 1 / (1+ np.exp(-pd.read_csv(open(utils.tmp_data_path + 'vwV12_{0}r{1}_test.txt_pred.txt'.format(is_ffm,i), 'r'), header=None).ix[:,0].values))
    ctr += 1
vw_pred /= ctr
print ("VW prediction loaded with shape", vw_pred.shape)

#factorization machine model output
ctr = 0
fm_pred = 0
for i in [51, 52, 53, 54]:
    fm_pred += pd.read_csv(open(utils.tmp_data_path + '{0}_r{1}_v.txt.out'.format(is_ffm,i), 'r'), header=None).ix[:,0].values
    ctr += 1
fm_pred /= ctr
print ("FM prediction loaded with shape", fm_pred.shape)

# 各模型的权重
blending_w = {'rf': .075, 'xgb': .175, 'vw': .225, 'fm': .525}

total_w = 0
pred = 0
# 预测*权重 累加
pred += rf_pred * blending_w['rf']
total_w += blending_w['rf']
pred += xgb_pred * blending_w['xgb']
total_w += blending_w['xgb']
pred += vw_pred * blending_w['vw']
total_w += blending_w['vw']
pred += fm_pred * blending_w['fm']
total_w += blending_w['fm']
# 平均的预测值
pred /= total_w

if utils.tvh == 'Y':
    #create submission
    predh_raw_avg = pred
    # 取出31日的site_id
    site_ids_h = site_id_values[day_values == 31] 
    # site_id 是否等于 '17d1b03f'
    tmp_f1 = site_ids_h == '17d1b03f'
    # site_id=='17d1b03f' 的预测概率 * (0.13/mean)
    predh_raw_avg[tmp_f1] *= .13 / predh_raw_avg[tmp_f1].mean()
    # 整个预测的概率 * (0.161/mean)
    predh_raw_avg *= .161 / predh_raw_avg.mean()

    # 打开sampleSubmission
    sub0 = pd.read_csv(open(utils.raw_data_path + 'sampleSubmission', 'r'))
    # 保留4位小数
    pred_h_str = ["%.4f" % x for x in predh_raw_avg]
    # 保存概率到 sub0
    sub0['click'] = pred_h_str
    fn_sub = utils.tmp_data_path + 'sub_sample' + str(utils.sample_pct) + '.csv'#'.csv.gz'
    import gzip
    # 保存 sub0 到 fn_sub
    sub0.to_csv(open(fn_sub, 'w'), index=False)
    print ("=" * 80)
    print ("Training complted and submission file " + fn_sub + " created.")
    print ("=" * 80)
else:
    #validate using day30
    # 30号 为校验集, 输出logloss
    print ("Training completed!")
    print ("=" * 80)
    print ("logloss of blended prediction:", logloss(pred, click_values[day_values==day_test]))
    print ("=" * 80)
