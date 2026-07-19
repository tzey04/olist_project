"""
Olist 电商数据分析 - 完整分析流程
============================================
包含: 数据加载与合并 -> 数据清洗 -> 探索性分析(EDA) -> RFM客户分层 -> 配送时效与满意度检验
运行方式: 在项目根目录下执行 `python scripts/full_analysis.py`
输出: 所有图表保存至 figures/ 目录
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

DATA_DIR = 'data'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)


def load_raw_tables():
    """读取9张原始csv表"""
    tables = {}
    filenames = {
        'orders': 'olist_orders_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'category_translation': 'product_category_name_translation.csv',
    }
    for key, fname in filenames.items():
        tables[key] = pd.read_csv(os.path.join(DATA_DIR, fname))
    return tables


def build_master_tables(t):
    """
    构建两张不同粒度的主表:
    - full: 商品级别(一行=一件商品), 用于销售/商品/卖家分析
    - orders_summary: 订单级别(一行=一个订单), 用于客户/支付/评价分析
    """
    # ---- full: 商品级别 ----
    full = t['orders'].merge(t['customers'], on='customer_id', how='left')
    full = full.merge(t['order_items'], on='order_id', how='left')
    full = full.merge(t['products'], on='product_id', how='left')
    full = full.merge(t['sellers'], on='seller_id', how='left')
    full = full.merge(t['category_translation'], on='product_category_name', how='left')

    # 日期转换
    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                 'order_estimated_delivery_date']
    for col in date_cols:
        full[col] = pd.to_datetime(full[col])
    full['delivery_days'] = (full['order_delivered_customer_date'] - full['order_purchase_timestamp']).dt.days

    # 缺失值处理: 类目名缺失填unknown(不是merge导致的缺失,是products表本身的缺失)
    full['product_category_name'] = full['product_category_name'].fillna('unknown')
    full['product_category_name_english'] = full['product_category_name_english'].fillna('unknown')

    # ---- orders_summary: 订单级别 ----
    payments_summary = t['payments'].groupby('order_id')['payment_value'].sum().reset_index()
    orders_summary = t['orders'].merge(t['customers'], on='customer_id', how='left')
    orders_summary = orders_summary.merge(t['reviews'], on='order_id', how='left')
    orders_summary = orders_summary.merge(payments_summary, on='order_id', how='left')
    for col in date_cols:
        orders_summary[col] = pd.to_datetime(orders_summary[col])
    orders_summary['delivery_days'] = (
        orders_summary['order_delivered_customer_date'] - orders_summary['order_purchase_timestamp']
    ).dt.days

    return full, orders_summary


def run_eda(full):
    """阶段3: 探索性分析,产出3张图"""
    # 3.1 月度销售趋势
    full['order_month'] = full['order_purchase_timestamp'].dt.to_period('M')
    monthly_sales = full.groupby('order_month')['price'].sum()
    monthly_sales = monthly_sales['2017-01':'2018-08']

    plt.figure(figsize=(10, 5))
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Monthly Sales Revenue (2017-01 to 2018-08)')
    plt.xlabel('Month')
    plt.ylabel('Revenue (BRL)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/monthly_sales.png', dpi=120)
    plt.close()

    # 3.2 品类销售额 Top10
    top_categories = full.groupby('product_category_name_english')['price'].sum() \
        .sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    top_categories.sort_values().plot(kind='barh', color='#2E86AB')
    plt.title('Top 10 Categories by Revenue')
    plt.xlabel('Revenue (BRL)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/top_categories.png', dpi=120)
    plt.close()

    # 3.3 各州销售额 Top10
    state_orders = full.groupby('customer_state').agg(
        order_count=('order_id', 'nunique'),
        revenue=('price', 'sum')
    ).sort_values('revenue', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    state_orders['revenue'].sort_values().plot(kind='barh', color='#A23B72')
    plt.title('Top 10 States by Revenue')
    plt.xlabel('Revenue (BRL)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/top_states.png', dpi=120)
    plt.close()

    return monthly_sales, top_categories, state_orders


def run_rfm(full):
    """阶段4.1: RFM客户分层, 产出1张图"""
    snapshot_date = full['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = full.groupby('customer_unique_id').agg(
        recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('price', 'sum')
    )
    rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['is_repeat'] = rfm['frequency'] > 1

    def assign_segment(row):
        if row['is_repeat']:
            return 'Loyal'
        elif row['R_score'] >= 4 and row['M_score'] >= 4:
            return 'Champions'
        elif row['R_score'] <= 2 and row['M_score'] >= 4:
            return 'At Risk'
        elif row['R_score'] <= 2 and row['M_score'] <= 2:
            return 'Lost'
        else:
            return 'Regular'

    rfm['segment'] = rfm.apply(assign_segment, axis=1)

    segment_summary = rfm.groupby('segment').agg(
        customer_count=('recency', 'count'),
        avg_monetary=('monetary', 'mean')
    ).sort_values('customer_count', ascending=False)

    plt.figure(figsize=(8, 5))
    rfm['segment'].value_counts().sort_values().plot(kind='barh', color='#F18F01')
    plt.title('Customer Segments (RFM)')
    plt.xlabel('Number of Customers')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/rfm_segments.png', dpi=120)
    plt.close()

    return rfm, segment_summary


def run_delivery_satisfaction_analysis(t):
    """阶段4.2: 配送时效与满意度关系, 产出1张图 + 假设检验结果"""
    orders = t['orders'].copy()
    reviews = t['reviews']
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days

    delivery_review = orders[['order_id', 'delivery_days']].merge(
        reviews[['order_id', 'review_score']], on='order_id', how='inner'
    )
    delivery_review = delivery_review.dropna(subset=['delivery_days', 'review_score'])

    summary = delivery_review.groupby('review_score')['delivery_days'].agg(['mean', 'median', 'count'])

    plt.figure(figsize=(8, 6))
    delivery_review.boxplot(column='delivery_days', by='review_score')
    plt.ylim(0, 60)
    plt.title('Delivery Days by Review Score')
    plt.suptitle('')
    plt.xlabel('Review Score')
    plt.ylabel('Delivery Days')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/delivery_vs_review.png', dpi=120)
    plt.close()

    rho, p_value = stats.spearmanr(delivery_review['delivery_days'], delivery_review['review_score'])

    return summary, rho, p_value


if __name__ == '__main__':
    print("加载原始数据...")
    tables = load_raw_tables()

    print("构建主表(full + orders_summary)...")
    full, orders_summary = build_master_tables(tables)
    print(f"  full: {full.shape}")
    print(f"  orders_summary: {orders_summary.shape}")

    print("\n运行EDA...")
    monthly_sales, top_categories, state_orders = run_eda(full)

    print("\n运行RFM客户分层...")
    rfm, segment_summary = run_rfm(full)
    print(segment_summary)

    print("\n运行配送时效与满意度分析...")
    delivery_summary, rho, p_value = run_delivery_satisfaction_analysis(tables)
    print(delivery_summary)
    print(f"Spearman rho = {rho:.4f}, p-value = {p_value:.2e}")

    print(f"\n全部完成,图表已保存至 {FIG_DIR}/ 目录")
