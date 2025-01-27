import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
try:
    medal_counts = pd.read_csv(r'D:\Education\2025 COMPA\2025_Problem_C_Data\summerOly_medal_counts.csv', encoding='utf-8')
except UnicodeDecodeError:
    medal_counts = pd.read_csv(r'D:\Education\2025 COMPA\2025_Problem_C_Data\summerOly_medal_counts.csv', encoding='latin1')

try:
    athletes = pd.read_csv(r'D:\Education\2025 COMPA\2025_Problem_C_Data\summerOly_athletes.csv', encoding='utf-8')
except UnicodeDecodeError:
    athletes = pd.read_csv(r'D:\Education\2025 COMPA\2025_Problem_C_Data\summerOly_athletes.csv', encoding='latin1')

try:
    hosts = pd.read_csv(r'D:\Education\2025 COMPA\2025_Problem_C_Data\summerOly_hosts.csv', encoding='utf-8')
except UnicodeDecodeError:
    hosts = pd.read_csv(r'D:\Education\2025 COMPA\2025_Problem_C_Data\summerOly_hosts.csv', encoding='latin1')

# 建立国家全称到简称的映射关系
country_mapping = athletes[['Team', 'NOC']].drop_duplicates()
country_mapping = country_mapping.dropna()  # 删除缺失值
country_mapping = country_mapping.drop_duplicates(subset=['Team', 'NOC'], keep='first')  # 删除重复项
full_to_short = country_mapping.set_index('Team')['NOC'].to_dict()  # 创建全称到简称的映射字典

# 将 medal_counts 中的国家全称替换为简称
medal_counts['NOC'] = medal_counts['NOC'].map(full_to_short)

# 数据清洗和预处理
hosts.columns = hosts.columns.str.strip()
hosts.rename(columns={'Host Country': 'Host'}, inplace=True)
medal_counts['Year'] = pd.to_numeric(medal_counts['Year'], errors='coerce')

# 从 athletes 数据集中提取队伍规模和首次参赛年份
team_size = athletes.groupby(['NOC', 'Year']).size().reset_index(name='Team Size')
first_participation = athletes.groupby('NOC')['Year'].min().reset_index(name='First Participation Year')

# 合并数据
medal_counts = medal_counts.merge(hosts, on='Year', how='left')
medal_counts = medal_counts.merge(team_size, on=['NOC', 'Year'], how='left')
medal_counts = medal_counts.merge(first_participation, on='NOC', how='left')

# 处理缺失值
medal_counts['First Participation Year'] = medal_counts['First Participation Year'].fillna(medal_counts['Year'])
medal_counts['Team Size'] = medal_counts['Team Size'].fillna(0)

# 创建新特征：当前年份与首次参赛年份的差值
medal_counts['Years Since First Participation'] = medal_counts['Year'] - medal_counts['First Participation Year']

# 添加历史成绩特征
medal_counts["Gold_h"] = medal_counts.groupby('NOC')['Gold'].shift(1).fillna(0)
medal_counts["Total_h"] = medal_counts.groupby('NOC')['Total'].shift(1).fillna(0)

# 添加是否为主办国的特征
medal_counts['is_host'] = (medal_counts['NOC'] == medal_counts['Host']).astype(int)

# 更新特征矩阵
features = ['Year', 'is_host', 'Gold_h', 'Total_h', 'Team Size', 'Years Since First Participation']
data = medal_counts[features + ['Gold', 'Silver', 'Bronze', 'Total', 'NOC']].dropna()

# 定义目标变量
targets = ['Gold', 'Silver', 'Bronze', 'Total']

# 存储每个目标变量的模型和预测结果
models = {}
predictions = {}

# 随机森林回归与超参数调优
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestRegressor(random_state=42)

for target in targets:
    y = data[target]
    X = data[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=2,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    models[target] = best_rf_model
    print(f"Best Parameters for {target}:", grid_search.best_params_)

    y_pred_rf = best_rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"{target} Optimized Random Forest Model MSE:", mse_rf)
    print(f"{target} Optimized Random Forest Model R2 Score:", r2_rf)

    predictions[target] = y_pred_rf

# 预测2028年奖牌数
# 获取所有国家的NOC代码
all_countries = data['NOC'].unique()

# 为每个国家创建2028年的预测数据
predictions_2028 = []

for country in all_countries:
    # 获取该国家在上一届奥运会（2024年）的队伍规模
    team_size_2024 = team_size[(team_size['NOC'] == country) & (team_size['Year'] == 2024)]['Team Size'].values
    if len(team_size_2024) == 0:  # 如果没有2024年的数据，则使用平均值
        team_size_2024 = team_size['Team Size'].mean()
    else:
        team_size_2024 = team_size_2024[0]

    # 获取该国家的首次参赛年份
    first_participation_year = first_participation[first_participation['NOC'] == country][
        'First Participation Year'].values
    if len(first_participation_year) == 0:  # 如果没有记录，则使用最早年份
        first_participation_year = data['Year'].min()
    else:
        first_participation_year = first_participation_year[0]

    # 创建预测数据
    example_2028 = pd.DataFrame({
        'Year': [2028],
        'is_host': [1 if country == 'USA' else 0],  # 假设2028年奥运会在美国举办
        'Gold_h': [data[(data['NOC'] == country) & (data['Year'] == 2024)]['Gold_h'].values[0] if len(
            data[(data['NOC'] == country) & (data['Year'] == 2024)]) > 0 else 0],
        'Total_h': [data[(data['NOC'] == country) & (data['Year'] == 2024)]['Total_h'].values[0] if len(
            data[(data['NOC'] == country) & (data['Year'] == 2024)]) > 0 else 0],
        'Team Size': [team_size_2024],
        'Years Since First Participation': [2028 - first_participation_year]
    })

    # 确保预测数据的列名和顺序与训练数据一致
    example_2028 = example_2028[features]

    # 预测该国家的奖牌数
    country_predictions = {target: models[target].predict(example_2028)[0] for target in targets}
    country_predictions['NOC'] = country
    predictions_2028.append(country_predictions)

# 将预测结果转换为DataFrame
predictions_df = pd.DataFrame(predictions_2028)

# 计算总奖牌数并进行排名
predictions_df['Total Predicted'] = predictions_df[['Gold', 'Silver', 'Bronze']].sum(axis=1)
predictions_df['Rank'] = predictions_df['Total Predicted'].rank(ascending=False, method='min').astype(int)

# 生成金牌排行榜
gold_medal_ranking = predictions_df[['NOC', 'Gold']].copy()
gold_medal_ranking['Gold Rank'] = gold_medal_ranking['Gold'].rank(ascending=False, method='min').astype(int)
gold_medal_ranking = gold_medal_ranking.sort_values(by='Gold Rank')

# 设置 Pandas 显示选项，确保输出时不省略数据
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 输出总奖牌数排行榜
print("\n总奖牌数排行榜：")
print(predictions_df.sort_values(by='Rank').reset_index(drop=True).to_string(index=False))

# 输出2028年奥运会金牌排行榜
print("\n2028年奥运会金牌排行榜：")
print(gold_medal_ranking.reset_index(drop=True).to_string(index=False))


# 预测特定国家的数据
def predict_specific_country(country_code):
    if country_code not in predictions_df['NOC'].values:
        print(f"国家代码 {country_code} 未找到，请检查输入是否正确。")
        return

    country_data = predictions_df[predictions_df['NOC'] == country_code].iloc[0]
    print(f"\n2028年奥运会预测数据 - {country_code}")
    print(f"金牌数: {country_data['Gold']:.2f}")
    print(f"银牌数: {country_data['Silver']:.2f}")
    print(f"铜牌数: {country_data['Bronze']:.2f}")
    print(f"总奖牌数: {country_data['Total Predicted']:.2f}")
    print(f"总奖牌排名: {country_data['Rank']}")
    print(f"金牌排名: {gold_medal_ranking[gold_medal_ranking['NOC'] == country_code]['Gold Rank'].values[0]}")


# 示例：预测特定国家的数据
predict_specific_country('CHN')  # 例如预测中国的数据




