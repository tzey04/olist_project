# Olist 巴西电商数据分析项目

基于 Olist（巴西最大电商平台之一）2016-2018 年的真实订单数据，完成从数据清洗、探索性分析到客户分层与统计检验的完整数据分析流程，产出可复现的分析代码与业务报告。

> 完整分析报告见 [`docs/report.md`](docs/report.md)（含全部图表与结论）

## 数据来源

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)（Kaggle 公开数据集），包含订单、客户、商品、卖家、支付、评价等 9 张关联表，约 10 万笔订单。

## 项目结构

```
olist_project/
├── data/                    # 原始数据(9张csv, 需自行从Kaggle下载)
├── scripts/
│   └── full_analysis.py     # 完整分析流程(一键复现全部结果)
├── notebooks/
│   └── 01_data_exploration.ipynb   # 分析过程notebook
├── figures/                 # 输出图表
├── docs/
│   └── report.md            # 完整分析报告(Markdown版)
├── reports/
│   └── Olist分析报告.docx    # 完整分析报告(Word版)
├── requirements.txt
└── README.md
```

## 技术栈

Python · pandas · matplotlib · scipy（假设检验）

## 如何运行

```bash
pip install -r requirements.txt
python scripts/full_analysis.py
```

运行后会在 `figures/` 目录生成全部图表，终端输出关键统计结果。

## 分析内容概览

| 模块 | 方法 | 关键发现 |
|---|---|---|
| 销售趋势 | 时间序列聚合 | 2017-2018年销售额增长约10倍,2017年11月出现明显峰值 |
| 品类与地域分布 | 分组聚合 | 美妆健康类目贡献最高营收;圣保罗州(SP)订单量是第二名的3倍以上 |
| 客户分层(RFM) | Recency/Frequency/Monetary 分位数打分 | 复购率仅 **3.12%**;"流失高价值客户"群体(At Risk)历史消费甚至高于活跃客户(Champions),是最具挖掘价值的召回目标 |
| 配送时效与满意度 | Spearman 等级相关检验 | 配送天数与评分呈统计显著负相关(rho=-0.23, p<0.001),1星评价订单平均配送时长(20.85天)是5星订单(10.22天)的约2倍 |

## 数据处理中值得记录的细节

- **多表合并按粒度分层管理**：区分"商品级别"(`full`)与"订单级别"(`orders_summary`)两张主表，避免因粒度不一致导致的行数膨胀和指标重复计算
- **识别数据集特有陷阱**：`customer_id` 实际上是"一次下单"的标识，同一用户复购会产生不同的 `customer_id`；必须使用 `customer_unique_id` 才能正确识别复购行为，否则会得出"复购率为0"的错误结论
- **缺失值按业务含义分类处理**：配送日期缺失保留为空值(代表未完成订单的真实状态)，商品类目缺失填充为"unknown"(避免误删有效销售记录)

## 后续可扩展方向

- 卖家维度的履约表现评估体系
- 基于文本的评价情感分析(NLP)
- 客户流失预测模型(结合RFM特征做监督学习)
