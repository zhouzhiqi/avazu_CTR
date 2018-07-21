1.参考1_FE_CNT_3.ipynb，将所有特征改名为C_(N)，并统计每个特征的取值，生成C_count_dict.json。
说明：我每个特征都是类别型，经过LabelEncode过（取值0~N）。Int类型对应修改。
生成：tr_50_FE.csv, va_50_FE_csv, ts_FE.csv

2.python run.py。生成ffm文件
需要注意的就是common.py文件中的特征总数，我是硬编码的18。
这段程序要并行处理需要在linux系统上（用到cat命令）。

3.训练
ffm-train -l 0.00001 -k 100 -t 40 -r 1 -s 12  --auto-stop  -p va_50_FE.ffm tr_50_FE.ffm 50_FE.model
ffm-predict ts_FE.csv 50_FE.model 50_FE.output