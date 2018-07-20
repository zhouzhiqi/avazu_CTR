import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import pylab 
import sys
import time
import os
import utils
from utils import *
from sklearn.externals.joblib import dump, load, Parallel, delayed

sys.path.append(utils.xgb_path)
import xgboost as xgb

# 准备好xgb参数
rseed = 0
xgb_eta = .3
tvh = utils.tvh
n_passes = 5
n_trees = 40
n_iter = 7
n_threads = 8
nr_factor = 4

# 寻找命令行中的参数及取值 
# n_passes和rseed
i = 1
while i < len(sys.argv):
    if sys.argv[i] == '-rseed':
        i += 1
        rseed = int(sys.argv[i])
    elif sys.argv[i] == '-passes':
        i += 1
        n_passes = int(sys.argv[i])
    else:
        raise ValueError("unrecognized parameter [" + sys.argv[i] + "]")
    
    i += 1

# FM模型参数
learning_rate = .1
# 数据路径
path1 = utils.tmp_data_path
param_names = '_r' + str(rseed)
# 生成训练集和校验集的文件名
#fn_t = path1 + 'fm_' + param_names + '_t.txt'
#fn_v = path1 + 'fm_' + param_names + '_v.txt'

# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
if utils.ffm: 
    is_ffm = 'FFM_'
else: 
    is_ffm = 'fm_'
fn_t = path1 + is_ffm + param_names + '_t.txt'
fn_v = path1 + is_ffm + param_names + '_v.txt'
fn_m = path1 + is_ffm + param_names + '_v.model'  
fn_o = path1 + is_ffm + param_names + '_v.txt.out' 
# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou

# 是否以31号为校验集
test_day = 30
if tvh == 'Y':
    test_day = 31

# 创建数据
def build_data():
    # 导入保存好的从t0中取出的43维特征值
    t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')
    # 取出对应数据
    t0tv_mx = t0tv_mx_save['t0tv_mx']
    click_values = t0tv_mx_save['click']
    day_values = t0tv_mx_save['day']

    print ("t0tv_mx loaded with shape ", t0tv_mx.shape)
    # 随机数种子
    np.random.seed(rseed)
    # 总数据量
    nn = t0tv_mx.shape[0]
    # 用于随机采样的id, 采样0.25
    r1 = np.random.uniform(0, 1, nn)

    # 22 <= ** < day_test, 再随机采样0.3
    filter1 = np.logical_and(np.logical_and(day_values >= 22, day_values < test_day), np.logical_and(r1 < 0.25, True))
    # 用于通过 日期 切分出 测试集
    filter_v1 = day_values == test_day
    # 训练用x,y
    xt1 = t0tv_mx[filter1, :]
    yt1 = click_values[filter1]
    # 判断之前生成的数据是否正确
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        print (xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    # 把训练数据, 转成xgb专用数据
    dtrain = xgb.DMatrix(xt1, label=yt1)
    # 把通过 日期 切分出 测试集, 转成xgb专用数据
    dvalid = xgb.DMatrix(t0tv_mx[filter_v1], label=click_values[filter_v1])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print (xt1.shape, yt1.shape)

    # 设置xgbboost参数
    param = {'max_depth':6, 'eta':.5, 'objective':'binary:logistic', 'verbose':0,
             'subsample':1.0, 'min_child_weight':50, 'gamma':0,
             'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': rseed}
    # 生成xgb所有参数
    plst = list(param.items()) + [('eval_metric', 'logloss')]
    # 开始训练xgboost
    xgb_test_basis_d6 = xgb.train(plst, dtrain, n_trees, watchlist)

    # 把总的数据, 转成xgb专用数据
    dtv = xgb.DMatrix(t0tv_mx)
    # 输出对应的叶子结点的id
    xgb_leaves = xgb_test_basis_d6.predict(dtv, pred_leaf = True)
    # 事先生成df用于保存xgb叶子结点的id
    t0 = pd.DataFrame({'click': click_values})
    print (xgb_leaves.shape)
    #逐一把树的叶子结点的id, 保存到df中
    for i in range(n_trees):
        # 用于保存的叶子结点的id
        pred2 = xgb_leaves[:, i]
        print (i, np.unique(pred2).size)
        # 用于保存的特征名
        t0['xgb_basis'+str(i)] = pred2

    # 导入事先保存的 对t0 LabelEncoder 后的结果
    t3a_save = load(utils.tmp_data_path + 't3a.joblib_dat')
    t3a = t3a_save['t3a']

    idx_base = 0
    for vn in ['xgb_basis' + str(i) for i in range(n_trees)]:
        # 对刚刚生成的xgb叶子结点的id进行LabelEncoder
        _cat = np.asarray(t0[vn].astype('category').values.codes, dtype='int32')
        # 加上起始的id号
        _cat1 = _cat + idx_base
        print (vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
        # 保存到t3a
        t3a[vn] = _cat1
        # 更新起始的id号
        idx_base += _cat.max() + 1
    #t3a.loc[np.logical_and(np.logical_and(day_values < test_day, day_values >= 22), True),:].to_csv(open(fn_t, 'w'), sep='\t', header=False, index=False)
    #t3a.loc[day_values==test_day,:].to_csv(open(fn_v, 'w'), sep='\t', header=False, index=False)

    # changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
    if utils.ffm: 
        # 训练数据: 保存几天前的结果([22,test_day)) 
        generate_ffm_data_zhou(t3a.loc[np.logical_and(np.logical_and(day_values < test_day, day_values >= 22), True),:], 'click', fn_t)
        # 校验数据: 保存当天的结果(test_day)
        generate_ffm_data_zhou(t3a.loc[day_values==test_day,:], 'click', fn_v)
    else:
        # 训练数据: 保存几天前的结果([22,test_day)) 
        generate_fm_data_zhou(t3a.loc[np.logical_and(np.logical_and(day_values < test_day, day_values >= 22), True),:], 'click', fn_t)
        # 校验数据: 保存当天的结果(test_day)
        generate_fm_data_zhou(t3a.loc[day_values==test_day,:], 'click', fn_v)

build_data()
import gc
gc.collect()


import os
#fm_cmd = utils.fm_path + ' -k ' + str(nr_factor) + ' -t ' + str(n_iter) + ' -s '+ str(n_threads) + ' '
#fm_cmd += ' -d ' + str(rseed) + ' -r ' + str(learning_rate) + ' ' + fn_v + ' ' + fn_t
#print (fm_cmd)
#os.system(fm_cmd)
#os.system("rm " + fn_t)
#os.system("rm " + fn_v)
# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
if utils.ffm: 
    fm_cmd = utils.ffm_path + 'ffm-train' + ' -k ' + str(nr_factor) + ' -t ' + str(n_iter) + ' -s '+ str(n_threads) + ' '
    fm_cmd += ' -r ' + str(learning_rate) + ' -p ' + fn_v + ' ' + fn_t + ' ' + fn_m
    fm_cmd_p = utils.ffm_path + 'ffm-predict' + ' ' + fn_v + ' ' + fn_m + ' ' + fn_o
else:
    fm_cmd = utils.fm_path + 'mf-train' + ' -k ' + str(nr_factor) + ' -t ' + str(n_iter) + ' -s '+ str(n_threads) + ' '
    fm_cmd += ' -f ' + str(5) + ' -r ' + str(learning_rate) + ' -p ' + fn_v + ' ' + fn_t + ' ' + fn_m
    fm_cmd_p = utils.fm_path + 'mf-predict' + ' -e ' + str(5) + ' ' + fn_v + ' ' + fn_m + ' ' + fn_o

print (fm_cmd)
os.system(fm_cmd)
print (fm_cmd_p)
os.system(fm_cmd_p)

# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
if utils.ffm: 
    pass
else:
    fm_pred = get_fm_predict_zhou(fn_v, fn_o)
    print('save fm pred ...', fm_pred.shape)
    pd.DataFrame(data=fm_pred).to_csv(fn_o,header=False,index=False)


for fn in [fn_v, fn_t, fn_m]:
    pass
    os.system("rm " + fn)
# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
