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


from sklearn.ensemble import RandomForestClassifier

# 是否以31号为校验集
day_test = 30
if utils.tvh == 'Y':
    day_test = 31

print ("to create Random Forest using day", day_test, " as validation")
# 实例化随机森林
clf = RandomForestClassifier(n_estimators=32, max_depth=40, min_samples_split=100, min_samples_leaf=10, random_state=0, criterion='entropy',
                             max_features=8, verbose = 1, n_jobs=-1, bootstrap=False)
# 历史记录的起始日期
_start_day = 22


predv = 0
ctr = 0
# 校验集的x,y
xv = t0tv_mx3[day_values==day_test, :]
yv = click_values[day_values==day_test]
# 总数据量
nn = t0tv_mx3.shape[0]


# 生成训练用数据, 进行多次训练, 
# 累加每次训练结果并取平均值, 再计算logloss
for i1 in range(8):
    # 修改RF的随机种子
    clf.random_state = i1
    np.random.seed(i1)
    r1 = np.random.uniform(0, 1, nn)  #用于随机采样
    # _start_day <= ** < day_test, 再随机采样0.3
    filter1 = np.logical_and(np.logical_and(day_values >= _start_day, day_values < day_test), np.logical_and(r1 < .3, True))
    # 训练用x,y
    xt1 = t0tv_mx3[filter1, :]
    yt1 = click_values[filter1]
    # 开始训练
    rf1 = clf.fit(xt1, yt1)
    # 开始预测
    y_hat = rf1.predict_proba(xv)[:, 1]
    # 分次累加预测结果
    predv += y_hat
    # 记录累加次数
    ctr += 1
    # 计算 多次 累加预测结果 的平均值的 logloss
    ll = logloss(predv/ctr, yv)
    print ("iter", i1, ", logloss = ", ll)
    sys.stdout.flush()


list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list = list_param + \
                            ['exptv_' + vn for vn in ['app_site_id', 'as_domain', 
                             'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                             'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']] + \
                            ['cnt_diff_device_ip_day_pday', 
                             'app_cnt_by_dev_ip', 'cnt_device_ip_day_hour', 'app_or_web',
                             'rank_dev_ip', 'rank_day_dev_ip', 'rank_app_dev_ip',
                             'diff_cnt_dev_ip_hour_phour_aw2_prev', 'diff_cnt_dev_ip_hour_phour_aw2_next',
                             'exp2_device_ip', 'exp2_app_site_id', 'exp2_device_model', 'exp2_app_site_model',
                             'exp2_app_site_model_aw', 'exp2_dev_ip_app_site',
                             'cnt_dev_ip', 'cnt_dev_id', 'hour1_web'] + \
                            ['all_withid', 'all_noid', 'all_but_ip', 'fm_5vars']
# 输出不同特征在RF中的重要性
print()
rf1_imp = pd.DataFrame({'feature':feature_list, 'impt': clf.feature_importances_})
print (rf1_imp.sort_values(by='impt', ascending=False))
print(rf1_imp.shape)
print ("to save validation predictions ...")
# 保存RF的最终预测结果(取平均后)
#dump(predv / ctr, utils.tmp_data_path + 'rf_pred_v.joblib_dat')
# changed by zhou
dump(predv / ctr, utils.tmp_data_path + '{0}rf_pred_v.joblib_dat'.format(is_ffm))  

