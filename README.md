# 项目 2: CTR 预估

# 问题描述

Click-Through Rate Prediction   
Predict whether a mobile ad will be clicked

In online advertising, click-through rate (CTR) is a very important metric for evaluating ad performance. As a result, click prediction systems are essential and widely used for sponsored search and real-time bidding.   
For this competition, we have provided 11 days worth of Avazu data to build and test prediction models. Can you find a strategy that beats standard classification algorithms? The winning models from this competition will be released under an open-source license.

# 评价指标

Submissions are evaluated using the `Logarithmic Loss` (smaller is better).   
`-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp)), yt为真实值, yp为预测概率`

# 提交文件格式

`最终输出的'click'为点击的` **`概率`**   
Submission Format
The submissions should contain the predicted probability of click for each ad impression in the test set using the following format:

id,click   
60000000,0.384   
63895816,0.5919   
759281658,0.1934   
895936184,0.9572   
......


# 特征含义

|名称|含义|
| :- | :- |
|id | ad identifier|
|click | 1是点击 0是末点击|
|hour | 时间, YYMMDDHH, so 14091123 means 2014年09月11号23:00 UTC.|
|C1 | 未知分类变量 |
|banner_pos|广告位置|
|site_id|网站ID号|
|site_domain|网站领域|
|site_category|网站类别|
|app_id|appID号|
|app_domain|应用领域|
|app_category|应用类别|
|device_id|设备ID号|
|device_ip|设备ip地址|
|device_model|设备型号, 如iphone5/4 等|
|device_type|设备类型, 如手机/平板电脑 等|
|device_conn_type|连接设备类型|
|C14-C21|未知分类变量 |


