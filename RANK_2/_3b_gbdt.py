import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
from sklearn.externals.joblib import dump, load, Parallel, delayed
import utils
from utils import *

# 把xgboost的路径添加到环境路径
sys.path.append(utils.xgb_path)
import xgboost as xgb

# changed by zhou
if utils.ffm: 
    is_ffm = 'FFM_'
else: 
    is_ffm = 'fm_'

# 导入保存好的从t0中取出的43维特征值与FM结果拼接在一起的数据
#t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx3.joblib_dat')
# changed by zhou
t0tv_mx_save = load(utils.tmp_data_path + '{0}t0tv_mx3.joblib_dat'.format(is_ffm))  # changed by zhou
t0tv_mx3 = t0tv_mx_save['t0tv_mx']
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
print ("t0tv_mx3 loaded with shape", t0tv_mx3.shape)

# 设置xgbboost参数
n_trees = utils.xgb_n_trees  #树的数量
# 是否以31号为校验集
day_test = 30
if utils.tvh == 'Y':
    day_test = 31
# 设置xgbboost参数
param = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}
# 总数据量
nn = t0tv_mx3.shape[0]
# 随机数种子
np.random.seed(999)
# 用于随机采样的id, 分成4份
sample_idx = np.random.random_integers(0, 3, nn)

# 生成训练用数据, 进行多次训练, 
# 累加每次训练结果并取平均值, 再计算logloss
predv_xgb = 0
ctr = 0
for idx in [0, 1, 2, 3]:
    # _start_day <= ** < day_test, 再随机采样1/4, 
    # 且每次用于训练的数据均不相同, 合并起来刚好是整个训练集
    filter1 = np.logical_and(np.logical_and(day_values >= 22, day_values < day_test), np.logical_and(sample_idx== idx , True))
    # 用于通过 日期 切分出 测试集
    filter_v1 = day_values == day_test
    # 切分出这次用于训练的数据
    xt1 = t0tv_mx3[filter1, :]
    yt1 = click_values[filter1]
    # 判断之前生成的数据是否正确
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        print (xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    # 1/4 训练数据, 转成xgb专用数据
    dtrain = xgb.DMatrix(xt1, label=yt1)
    # day_values == day_test 测试集数据, 转成xgb专用数据
    dvalid = xgb.DMatrix(t0tv_mx3[filter_v1], label=click_values[filter_v1])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print (xt1.shape, yt1.shape)
    # 生成xgb所有参数
    plst = list(param.items()) + [('eval_metric', 'logloss')]
    # 开始训练xgboost
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)
    #xgb_pred[rseed] = xgb1.predict(dtv3)
    #xgb_list[rseed] = xgb1

    # 记录累加次数
    ctr += 1
    # 分次累加预测结果
    predv_xgb += xgb1.predict(dvalid)
    # 计算 多次 累加预测结果 的平均值的 logloss
    print ('-'*30, ctr, logloss(predv_xgb / ctr, click_values[filter_v1]))

# 保存GBDT的最终预测结果(取平均后)
print ("to save validation predictions ...")
dump(predv_xgb / ctr, utils.tmp_data_path + '{0}xgb_pred_v.joblib_dat'.format(is_ffm))

