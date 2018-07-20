import numpy as np
import pandas as pd
from sklearn.utils import check_random_state 
import time
import sys
from sklearn.externals.joblib import dump, load
import matplotlib.pyplot as plt

############ changed by zhou#####################################
dtypes = {'id':np.uint64,'click':np.int8,'hour':np.int64,'C1':np.int32,'banner_pos':np.int32,
         'device_conn_type':np.int32,'device_type':np.int32,'C14':np.int32,'C15':np.int32,
         'C16':np.int32,'C17':np.int32,'C18':np.int32,'C19':np.int32,'C20':np.int32,'C21':np.int32,
         }
########### changed by zhou######################################
sample_pct = .5
tvh = 'N'  #以31号做为校验集， 用于生成最终预测
xgb_n_trees = 300
xgb_n_trees = 30 # changed by zhou
ffm = True  # changed by zhou


#Please set following path accordingly
##############################
try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.raw_data_path = '../data/'
            self.tmp_data_path = '../tmp_data/'
            self.train_name = 'minitrain'
            self.test_name = 'test'
#实例化class
FLAGS = flags()
###############################
#where we can find training, test, and sampleSubmission.csv
raw_data_path = FLAGS.raw_data_path
#where we store results -- require about 130GB
tmp_data_path = FLAGS.tmp_data_path
train_name = FLAGS.train_name
test_name = FLAGS.test_name

#path to external binaries. Please see dependencies in the .pdf document
fm_path = '../tools/libmf_2.01/'
ffm_path = '../tools/libffm/'
xgb_path = '../tools/libmf_2.01/'
vw_path = '../tools/vw_2017/vw '


try:
    params=load(tmp_data_path + '_params.joblib_dat')
    sample_pct = params['pct']
    tvh = params['tvh']
except:
    pass


def print_help():
    print( "usage: python utils -set_params [tvh=Y|N], [sample_pct]")
    print( "for example: python utils -set_params N 0.05")

def main():
    if sys.argv[1] == '-set_params' and len(sys.argv) == 4:
        try:
            tvh = sys.argv[2]
            sample_pct = float(sys.argv[3])
            dump({'pct': sample_pct, 'tvh':tvh}, tmp_data_path + '_params.joblib_dat')
        except:
            print_help()
    else:
        print_help()

if __name__ == "__main__":
    main()

def get_agg(group_by, value, func):
    # 以 group_by 的不同取值 对 value 进行分组
    g1 = pd.Series(value).groupby(group_by)
    # 对分组好的数据, 执行 func
    agg1  = g1.aggregate(func)
    #print( agg1)
    # 仍以group_by 扩展执行好的结果
    r1 = agg1[group_by].values
    return r1

def calcLeaveOneOut2(df, vn, vn_y, cred_k, r_k, power, mean0=None, add_count=False):
    if mean0 is None:  #生成大量平均点击率
        mean0 = df[vn_y].mean() * np.ones(df.shape[0])
    # 对vn进行LabelEncodder
    _key_codes = df[vn].values.codes
    # 按 _key_codes 对 点击=0/1 分组
    grp1 = df[vn_y].groupby(_key_codes)
    # 按 _key_codes 对 平均点击次数 分组
    grp_mean = pd.Series(mean0).groupby(_key_codes)
    # 获取不同 _key_codes 的 平均点击次数(都一样)
    mean1 = grp_mean.aggregate(np.mean)
    # 获取不同 _key_codes 的 点击次数
    sum1 = grp1.aggregate(np.sum)
    # 获取不同 _key_codes 的 浏览次数
    cnt1 = grp1.aggregate(np.size)
    
    #print (sum1)
    #print (cnt1)
    vn_sum = 'sum_' + vn
    vn_cnt = 'cnt_' + vn
    # 把 点击次数, 浏览次数, 平均点击次数 展开
    _sum = sum1[_key_codes].values
    _cnt = cnt1[_key_codes].values
    _mean = mean1[_key_codes].values
    #print (np.isnan(_sum).sum())
    #print (_cnt[:10])
    #print (_mean[:10])
    # 不在_key_codes中的值自动变成nan, 将其置0
    _mean[np.isnan(_sum)] = mean0.mean()
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    #print (_cnt[:10])
    # 减少一次点击次数
    _sum -= df[vn_y].values
    # 减少一次浏览次数
    _cnt -= 1
    #print (_cnt[:10])
    # 生成新的特征名子, 并判断是否添加
    vn_yexp = 'exp2_'+vn
    #df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    # _sum/_cnt 点击概率
    # 对调整后的点击率取power次方
    diff = np.power((_sum + cred_k * _mean)/(_cnt + cred_k) / _mean, power)
    # 如果新特征已经存在, 就直接乘以调整后的点击率
    if vn_yexp in df.columns:
        df[vn_yexp] *= diff
    # 如果新特征不存在, 就直接赋值调整后的点击率
    else:
        df[vn_yexp] = diff 
    # 如果r_k > 0, 继续调整点击率
    if r_k > 0:
        df[vn_yexp] *= np.exp((np.random.rand(np.sum(filter_train))-.5) * r_k)
    # 添加浏览次数
    if add_count:
        df[vn_cnt] = _cnt
    return diff


def my_lift(order_by, p, y, w, n_rank, dual_axis=False, random_state=0, dither=1e-5, fig_size=None):
    gen = check_random_state(random_state)
    if w is None:
        w = np.ones(order_by.shape[0])
    if p is None:
        p = order_by
    ord_idx = np.argsort(order_by + dither*np.random.uniform(-1.0, 1.0, order_by.size))
    p2 = p[ord_idx]
    y2 = y[ord_idx]
    w2 = w[ord_idx]

    cumm_w = np.cumsum(w2)
    total_w = cumm_w[-1]
    r1 = np.minimum(n_rank, np.maximum(1, 
                    np.round(cumm_w * n_rank / total_w + .4999999)))
    
    df1 = pd.DataFrame({'r': r1, 'pw': p2 * w2, 'yw': y2 * w2, 'w': w2})
    grp1 = df1.groupby('r')
    
    sum_w = grp1['w'].aggregate(np.sum)
    avg_p = grp1['pw'].aggregate(np.sum) / sum_w 
    avg_y = grp1['yw'].aggregate(np.sum) / sum_w
    
    xs = range(1, n_rank+1)
    
    fig, ax1 = plt.subplots()
    if fig_size is None:
        fig.set_size_inches(20, 15)
    else:
        fig.set_size_inches(fig_size)
    ax1.plot(xs, avg_p, 'b--')
    if dual_axis:
        ax2 = ax1.twinx()
        ax2.plot(xs, avg_y, 'r')
    else:
        ax1.plot(xs, avg_y, 'r')
    
    #print( "logloss: ", logloss(p, y, w))
    
    return gini_norm(order_by, y, w)

def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)

def gini_norm(pred, y, weight=None):

    #equal weight by default
    if weight == None:
        weight = np.ones(y.size)

    #sort actual by prediction
    # 对预测结果排序
    ord = np.argsort(pred)
    # 生成排序后的结果
    y2 = y[ord]
    # 生成排序后的权重矩阵
    w2 = weight[ord]
    #gini by pred
    cumm_y = np.cumsum(y2)
    total_y = cumm_y[-1]
    total_w = np.sum(w2)
    g1 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    #sort actual by actual
    # 对真实值排序
    ord = np.argsort(y)
    # 生成排序后的结果
    y2 = y[ord]
    # 生成排序后的权重矩阵
    w2 = weight[ord]
    #gini by actual
    cumm_y = np.cumsum(y2)
    g0 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    return g1/g0

def mergeLeaveOneOut2(df, dfv, vn):
    # 对vn进行LabelEncodder
    _key_codes = df[vn].values.codes
    # 生成新的特征名子
    vn_yexp = 'exp2_'+vn
    # 按 _key_codes 对 vn_yexp 分组
    grp1 = df[vn_yexp].groupby(_key_codes)
    # 获取不同 _key_codes 的 平均点击率
    _mean1 = grp1.aggregate(np.mean)
    # 对dfv中的vn进行LabelEncodder, 
    # 并按新的LabelEncodder展开
    _mean = _mean1[dfv[vn].values.codes].values
    # 对按新的LabelEncodder展开的分组中的nan, 
    # 增补为老的LabelEncodder展开的总的均值
    _mean[np.isnan(_mean)] = _mean1.mean()

    return _mean
    
    
def calcTVTransform(df, vn, vn_y, cred_k, filter_train, mean0=None):
    if mean0 is None:
        # 前面几天/前一天 的平均点击率
        mean0 = df.loc[filter_train, vn_y].mean()
        print( "mean0:", mean0)
    else:
        # 今天 的平均点击率
        mean0 = mean0[~filter_train]
        
    # df['_key1'] = df[vn]进行LabelEncoder
    df['_key1'] = df[vn].astype('category').values.codes
    # 取出[前几天, ['_key1', vn_y]]
    df_yt = df.loc[filter_train, ['_key1', vn_y]]
    #df_y.set_index([')key1'])
    # 以'_key1'为index对数据进行分组
    grp1 = df_yt.groupby(['_key1'])
    # 对分组好的数据 统计 前几天 点击次数(vn_y的 sum)
    sum1 = grp1[vn_y].aggregate(np.sum)
    # 对分组好的数据 统计 前几天 浏览次数(vn_y的 size)
    cnt1 = grp1[vn_y].aggregate(np.size)
    
    vn_sum = 'sum_' + vn
    vn_cnt = 'cnt_' + vn
    
    # 取出[今天, '_key1']
    v_codes = df.loc[~filter_train, '_key1']
    # 把 前几天 的分组后的统计结果, 放到 今天, 
    # 如果今天的'_key1'的取值不在前几天中, 自动变为nan
    _sum = sum1[v_codes].values
    _cnt = cnt1[v_codes].values
    # 如果今天的'_key1'的取值不在前几天中, 重新赋值为0
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    
    r = {}
    # _sum/_cnt 今天的'_key1', 在前几天的点击概率, 
    # 下面为调整后的点击率
    r['exp'] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    r['cnt'] = _cnt

    return r

def cntDualKey(df, vn, vn2, key_src, key_tgt, fill_na=False):
    
    # 把 vn 分别与key_src, key_tgt,以str相加
    # 把 device_ip 分别与当前小时, 其它小时,以str相加
    print( "build src key")
    _key_src = np.add(df[key_src].astype('str').values, df[vn].astype('str').values)
    print( "build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('str').values, df[vn].astype('str').values)
    # 如果vn2不为空, 把 vn 分别与key_src, key_tgt,以str相加, 再加上 vn2
    # 如果vn2不为空, 把 device_ip 分别与当前小时, 其它小时,以str相加, 再加上 vn2
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('str').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('str').values)

    print( "aggreate by src key")
    # 对 df 以 _key_src 进行分组
    # 对 df 以 当前小时 进行分组
    grp1 = df.groupby(_key_src)
    # 以 当前小时 分组后的数据, 统计 不同device_ip 出现的次数
    cnt1 = grp1[vn].aggregate(np.size)
    
    print( "map to tgt key")
    # 以 当前小时 分组后的不同device_ip 出现的次数, 扩展到其它小时
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _cnt = cnt1[_key_tgt].values

    # 对扩展后出现的nan赋值
    if fill_na is not None:
        print( "fill in na")
        _cnt[np.isnan(_cnt)] = fill_na    
    # 生成新的特征名
    vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
    # 如果有vn2, 新的特征名添加vn2
    if vn2 is not None:
        vn_cnt_tgt += '_' + vn2
    # 添加新的特征名及对应数据
    df[vn_cnt_tgt] = _cnt

def my_grp_cnt(group_by, count_by):
    _ts = time.time()
    # 优先对group_by排序(从小到大), 并返回排序后的索引值,
    # 当group_by相等时, 再以对应索引的count_by排序
    _ord = np.lexsort((count_by, group_by))
    print (time.time() - _ts)
    _ts = time.time()    
    # 生成全是1的向量
    _ones = pd.Series(np.ones(group_by.size))
    print (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in range(1, group_by.size):
        i0 = _ord[i]  #最小的group_by的id
        # group_by的值 等于 '___'
        if _prev_grp == group_by[i0]:
            # 属于同一group_by的值, 不同的count_by的个数
            # 类似于以group_by分组, 计数不同的count_by的个数
            # 当前值与前一值不同时, running_cnt+1
            if count_by[_ord[i-1]] != count_by[i0]: 
                running_cnt += 1
        else:  # group_by的值 不等于 '___', 
            running_cnt = 1  #计数器置1
            _prev_grp = group_by[i0]  #更新待计数的值
        # 倒数第二条数据 或 当前排序好的group_by某一值计数完成
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            # 以group_by为模版, 对相同索引的_cs1, 
            # 写入计数好的count_by不同值的个数
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    print (time.time() - _ts)
    if True:
        return _cs1
    else:
        _ts = time.time()    

        org_idx = np.zeros(group_by.size, dtype=np.int)
        print (time.time() - _ts)
        _ts = time.time()    
        org_idx[_ord] = np.asarray(range(group_by.size))
        print (time.time() - _ts)
        _ts = time.time()    

        return _cs1[org_idx]
    
def my_cnt(group_by):
    _ts = time.time()
    _ord = np.argsort(group_by)
    print (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            running_cnt += 1
        else:
            running_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    print (time.time() - _ts)
    return _cs1

def my_grp_value_diff(group_by, order_by, value):
    _ts = time.time()
    _ord = np.lexsort((order_by, group_by))
    print (time.time() - _ts)
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i0] = value[_ord[i]] - value[_ord[i-1]]
        else:
            _cs1[i0] = 1e7
            _prev_grp = group_by[i0]
    print (time.time() - _ts)
    
    return np.minimum(_cs1, 1e7)

def my_grp_idx(group_by, order_by):
    _ts = time.time()
    # 优先对group_by排序(从小到大), 并返回排序后的索引值,
    # 当group_by相等时, 再以对应索引的count_by排序
    _ord = np.lexsort((order_by, group_by))
    print (time.time() - _ts)
    _ts = time.time() 
    # 生成全是1的向量
    _ones = pd.Series(np.ones(group_by.size))
    print (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    # 生成0向量
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in range(1, group_by.size):
        i0 = _ord[i]  #最小的group_by的id
        # group_by的值 等于 '___'
        if _prev_grp == group_by[i0]:
            # 属于同一group_by的值, 
            # 对排序后的count_by进行LabelEncoder
            _cs1[i] = _cs1[i - 1] + 1
        else:  #group_by的值换新的了
            _cs1[i] = 1  #索引重新置1
            # 更新下一group_by的值
            _prev_grp = group_by[i0]
    print (time.time() - _ts)
    _ts = time.time()    
    
    org_idx = np.zeros(group_by.size, dtype=np.int)
    print (time.time() - _ts)
    _ts = time.time()    
    # 以group_by为模版, 对相同索引的org_idx, 
    # 重新赋值range(group_by.size), 做为新的group_by的索引
    org_idx[_ord] = np.asarray(range(group_by.size))
    print (time.time() - _ts)
    _ts = time.time()    
    # 以原本group_by的顺序 返回重新定义的count_by的LabelEncoder
    return _cs1[org_idx]

def calcDualKey(df, vn, vn2, key_src, key_tgt, vn_y, cred_k, mean0=None, add_count=False, fill_na=False):
    # 总的点击概率
    if mean0 is None:
        mean0 = df[vn_y].mean()
    # 把 vn 分别与key_src, key_tgt,以str相加
    # 把 device_ip 分别与当前日期, 前一天,以str相加
    print ("build src key")
    _key_src = np.add(df[key_src].astype('str').values, df[vn].astype('str').values)
    print ("build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('str').values, df[vn].astype('str').values)
    # 如果vn2不为空, 把 vn 分别与key_src, key_tgt,以str相加, 再加上 vn2
    # 如果vn2不为空, 把 device_ip 分别与当前日期, 前一天,,以str相加, 再加上 vn2
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('str').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('str').values)

    print ("aggreate by src key")
    # 对 df 以 _key_src 进行分组
    # 对 df 以 当前日期 进行分组
    grp1 = df.groupby(_key_src)
    # 以 当前日期 分组后的数据, 统计 不同device_ip 的点击次数
    sum1 = grp1[vn_y].aggregate(np.sum)
    # 以 当前日期 分组后的数据, 统计 不同device_ip 的浏览次数
    cnt1 = grp1[vn_y].aggregate(np.size)
    
    print ("map to tgt key")
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    # 以 当前日期 分组后的不同device_ip 的点击次数, 扩展到其它小时
    _sum = sum1[_key_tgt].values
    # 以 当前日期 分组后的不同device_ip 的浏览次数, 扩展到其它小时
    _cnt = cnt1[_key_tgt].values

    if fill_na:
        print ("fill in na")
        # 对扩展后出现的nan 重新赋值
        _cnt[np.isnan(_sum)] = 0    
        _sum[np.isnan(_sum)] = 0

    print ("calc exp")
    # 如果有vn2, 新的特征名添加vn2
    if vn2 is not None:
        vn_yexp = 'exp_' + vn + '_' + vn2 + '_' + key_src + '_' + key_tgt
    else:
        vn_yexp = 'exp_' + vn + '_' + key_src + '_' + key_tgt
    # 添加新的特征名及对应的点击概率
    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)

    if add_count:
        print ("add counts")
        # 添加浏览次数
        vn_cnt_src = 'cnt_' + vn + '_' + key_src
        # 生成新特征名
        df[vn_cnt_src] = _cnt
        # 对 df 以 _key_tgt 进行分组
        # 对 df 以 前一天 进行分组
        grp2 = df.groupby(_key_tgt)
        # 对以 _key_tgt 进行分组后的数据, 统计click的浏览次数
        cnt2 = grp2[vn_y].aggregate(np.size)
        # 把统计好的数据直接展开
        _cnt2 = cnt2[_key_tgt].values
        vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
        # 添加新的特征名及对应的浏览次数
        df[vn_cnt_tgt] = _cnt2

def get_set_diff(df, vn, f1, f2):
    #print(df[vn].values.sum())
    set1 = set(np.unique(df[vn].values[f1]))
    set2 = set(np.unique(df[vn].values[f2]))
    set2_1 = set2 - set1
    print( vn, '\t', len(set1), '\t', len(set2), '\t', len(set2_1))
    return len(set2_1) * 1.0 / len(set2)

def calc_exptv(t0, vn_list, last_day_only=False, add_count=False):
    t0a = t0.loc[:, ['day', 'click']].copy()
    day_exps = {}

    #对特征进行组合, 并进行LabelEncoder
    for vn in vn_list:
        if vn == 'dev_id_ip':
            t0a[vn] = pd.Series(np.add(t0.device_id.values , t0.device_ip.values)).astype('category').values.codes
        elif vn == 'dev_ip_aw':
            t0a[vn] = pd.Series(np.add(t0.device_ip.values , t0.app_or_web.astype('str').values)).astype('category').values.codes
        elif vn == 'C14_aw':
            t0a[vn] = pd.Series(np.add(t0.C14.astype('str').values , t0.app_or_web.astype('str').values)).astype('category').values.codes
        elif vn == 'C17_aw':
            t0a[vn] = pd.Series(np.add(t0.C17.astype('str').values , t0.app_or_web.astype('str').values)).astype('category').values.codes
        elif vn == 'C21_aw':
            t0a[vn] = pd.Series(np.add(t0.C21.astype('str').values , t0.app_or_web.astype('str').values)).astype('category').values.codes
        elif vn == 'as_domain':
            t0a[vn] = pd.Series(np.add(t0.app_domain.values , t0.site_domain.values)).astype('category').values.codes
        elif vn == 'site_app_id':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.app_id.values)).astype('category').values.codes
        elif vn == 'app_model':
            t0a[vn] = pd.Series(np.add(t0.app_id.values , t0.device_model.values)).astype('category').values.codes
        elif vn == 'app_site_model':
            t0a[vn] = pd.Series(np.add(t0.app_id.values , np.add(t0.site_id.values , t0.device_model.values))).astype('category').values.codes
        elif vn == 'site_model':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.device_model.values)).astype('category').values.codes
        elif vn == 'app_site':
            t0a[vn] = pd.Series(np.add(t0.app_id.values , t0.site_id.values)).astype('category').values.codes
        elif vn == 'site_ip':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.device_ip.values)).astype('category').values.codes
        elif vn == 'app_ip':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.device_ip.values)).astype('category').values.codes
        elif vn == 'site_id_domain':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.site_domain.values)).astype('category').values.codes
        elif vn == 'site_hour':
            t0a[vn] = pd.Series(np.add(t0.site_domain.values , (t0.hour.values % 100).astype('str'))).astype('category').values.codes
        else:
            t0a[vn] = t0[vn]

        # 获取新特征前几日累加的点击概率与浏览次数
        for day_v in range(22, 32):
            cred_k = 10
            # 如果当前日期没有处理(不在day_exps), 创建空字典
            if day_v not in day_exps:
                day_exps[day_v] = {}

            vn_key = vn

            import time
            _tstart = time.time()

            day1 = 20
            # 前几天/前一天
            if last_day_only:
                day1 = day_v - 2
            # 过滤前几天/前一天 + 当天
            filter_t = np.logical_and(t0.day.values > day1, t0.day.values <= day_v)
            vn_key = vn
            t1 = t0a.loc[filter_t, :].copy()
            # 过滤前几天/前一天
            filter_t2 = np.logical_and(t1.day.values != day_v, t1.day.values < 31)
            # 获取前几天/前一天某特征某取值的点击概率与浏览次数
            if vn == 'app_or_web':
                day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'click', cred_k, filter_t2)
            else:
                if last_day_only:
                    day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'click', cred_k, filter_t2, mean0=t0[filter_t].expld_app_or_web.values)
                else:
                    day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'click', cred_k, filter_t2, mean0=t0[filter_t].exptv_app_or_web.values)
            
            print( vn, vn_key, " ", day_v, " done in ", time.time() - _tstart)
        t0a.drop(vn, inplace=True, axis=1)
    
    # 把统计好的结果, 放入到t0中
    for vn in vn_list:
        vn_key = vn
            
        # 列名: 前几天exptv_ / 前一天expld_ 
        vn_exp = 'exptv_'+vn_key
        if last_day_only:
            vn_exp='expld_'+vn_key
        # 列名: 是否添加浏览次数
        t0[vn_exp] = np.zeros(t0.shape[0])
        if add_count:
            t0['cnttv_'+vn_key] = np.zeros(t0.shape[0])
            
        # 添加提前命名好的 前几天/前一天 的 点击概率(和浏览次数)
        for day_v in range(22, 32):
            print( vn, vn_key, day_v, t0.loc[t0.day.values == day_v, vn_exp].values.size, day_exps[day_v][vn_key]['exp'].size)
            t0.loc[t0.day.values == day_v, vn_exp]=day_exps[day_v][vn_key]['exp']
            if add_count:
                t0.loc[t0.day.values == day_v, 'cnttv_'+vn_key]=day_exps[day_v][vn_key]['cnt']
        
# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
def generate_fm_data_zhou(df, target, file_name):
    df.loc[df[target]==0,target] = -1
    print(df.shape, 'to FM data format')
    # 生成空白文件
    with open(file_name,'w') as f: pass
    for c in df.columns:
        if c == target: continue
        #print(c)
        # 取出columns和targt, 
        df_tmp = df.loc[:,[c,target]]
        # 生成新的index
        df_tmp.index = np.arange(df.shape[0], dtype=np.int32)
        df_tmp.to_csv(file_name, mode='a',header=False, sep=' ',index=True)

# changed by zhou
def get_fm_predict_zhou(input_name, output_name):
    # 导入输入文件中的行索引
    index = pd.read_csv(input_name, usecols=[0], names=['index'], sep=' ')
    # 导入输出文件中的预测值 
    predict = pd.read_csv(output_name, usecols=[0], names=['click'])
    # 归一化预测值
    max = predict['click'].max()
    min = predict['click'].min()
    predict = (predict['click']-min)/(max-min)
    # 拼接在一起
    predict = pd.concat((predict, index),axis=1)
    # 以 index 分组, 统计 click 的平均值
    click_rate = predict.groupby('index').aggregate(np.mean)['click']
    return click_rate.values

# changed by zhou
def generate_ffm_data_zhou(df, target, file_name):
    fields = pd.Series(data=np.arange(df.shape[1]).astype(np.str), index=df.columns)
    print(df.shape, 'to FFM data format')
    # 生成空白文件
    for c in df.columns:
        if c == target: continue
        df.loc[:,c] = df.loc[:,c].map(lambda x:fields[c]+':'+str(x)+':1').values
    df.to_csv(file_name, mode='w',header=False, sep=' ',index=False)


# changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou changed by zhou
