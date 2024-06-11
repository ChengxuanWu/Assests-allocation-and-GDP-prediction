import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import reduce
from self_func import Stockassests, Datasets, get_engine


#Loading all the potential data through api, then transfer to Dataframe
#Class Datasets was defined above, for translate the data from 5 layers of json
GDP = Datasets('a018101010/1+3...Q.&.&startTime=1999-Q4&endTime=2024-Q1')
data_GDP = GDP.get_data()
time_GDP = GDP.get_time()
#add the latest data manually since the government doesn't update the database
lst_momenten = []
for i in range(1,len(data_GDP)):
    momenten = (data_GDP.iloc[i,1]-data_GDP.iloc[i-1,1])
    lst_momenten.append(momenten)
lst_momenten.insert(0, 0)
data_GDP['momentum'] = lst_momenten
del lst_momenten
#Add data for classifiction for circular economy
lst_limits = []
for i, j in zip(data_GDP.iloc[:,1], data_GDP.iloc[:,2]):
    if i > 0:
        if j >0:
            lst_limits.append(1)
        else:
            lst_limits.append(2)
    else:
        if j >0:
            lst_limits.append(4)
        else:
            lst_limits.append(3)
data_GDP['limits'] = lst_limits
del lst_limits
data_GDP.drop(0, axis=0, inplace=True)
time_GDP.drop(0, axis=0, inplace=True)
time_GDP['time'] = time_GDP['year']+' '+time_GDP['mon']
data_GDP['time'] = time_GDP['time']
#First, we have to show the trend of GDP growth rate through time
#below baseline: recession and recover, above baseline: prosper and slow growth
plt.figure(figsize=(10, 6))
x= range(97)
y= data_GDP.iloc[:,1]
plt.plot(x, y, marker='o')
time_zone = GDP.get_timezone()
plt.xticks(range(98), time_zone)
location = plt.gca()
x_major_locator = plt.MultipleLocator(12)
location.xaxis.set_major_locator(x_major_locator)
plt.ylabel('Real GDP Growth Rate (%)')
plt.xlabel('Time (yrs-quarter)')
plt.title('Real GDP Growth Rate', fontsize=24)
plt.axhline(0, ls='dashed', alpha=0.8, color='#FFA07A', linewidth=4)
threshold = 0
plt.fill_between(x, -8.5, 13, color='#05FF32', alpha=0.1, label="recession/recover")
plt.fill_between(x, -8.5, 13, where=y >= threshold, color='#F87646', alpha=0.2, label="growth")
plt.legend()
plt.ylim(-8.5,13)
plt.grid(True)
plt.show()

#The limits and depict the circular economic plot
#Get the circular plot to determine the four limit: prosper, steady, recession, recovery
#Using the GDP growth rate and momentum calculated by subtracting the value of previous quarter
plt.plot([-10, 15], [0, 0], ls='dashed', alpha=0.8, color='#778899')
plt.plot([0, 0], [-10, 15], ls='dashed', alpha=0.8, color='#778899')
plt.scatter(data_GDP['momentum'][:93], data_GDP.iloc[:93,1], c='#808080', edgecolors="#000000")
plt.xlim(min(data_GDP['momentum'])-2,max(data_GDP['momentum'])+2)
plt.ylim(min(data_GDP.iloc[:,1])-2,max(data_GDP.iloc[:,1])+2)
plt.scatter(data_GDP['momentum'][93:], data_GDP.iloc[93:,1], c='#dcf346', edgecolors='#9e1500', label="recent 4 quarter")
plt.title('Economic status', fontsize=20)
plt.ylabel('Real GDP Growth Rate (%)')
plt.xlabel('Momentum of growth (%)')
plt.text(7, 12, "Prosperity", fontsize=14)
plt.text(7, -8.5, "Recovery", fontsize=14)
plt.text(-8.5, -8.5, "Recession", fontsize=14)
plt.text(-8.5, 12, "Slow Growth", fontsize=14)
plt.legend(loc=(0.6, 0.2))
plt.show

#Loading stock infomation over 5 years
data_0050 = Stockassests('0050.TW', '2019-01-01', '2024-05-28').get_data()
data_00679B = Stockassests('00679B.TWO', '2019-01-01', '2024-05-28').get_data()
data_00713 = Stockassests('00713.TW', '2019-01-01', '2024-05-28').get_data()
stock = pd.concat([data_0050, data_00679B, data_00713], axis=1, join='inner')
stock.columns = ['0050', '00679B', '00713']
#export data to mysql-macro.db
engine = get_engine()
stock.to_sql('stocks', engine, if_exists='replace')

total_stocks = len(stock.columns)
#add limits to stock see the performances in different economic status
stock_time = pd.DataFrame(list(map(lambda x: str(x).split()[0].split('-'), stock.index)), 
                          dtype=float, columns=['Y', 'M', 'D'])
stock_time['Q'] = np.ceil((stock_time['M']/3))
limits_with_time = pd.DataFrame(data_GDP['limits'])
limits_with_time['Y'] = list(map(lambda x: str(x).split()[0], data_GDP['time'][:]))
limits_with_time['Q'] = list(map(lambda x: str(x).split()[1].lstrip('Q'), data_GDP['time'][:]))
limits_with_time = limits_with_time.astype(float)
stock_time = pd.merge(left=stock_time, right=limits_with_time, left_on=['Y','Q'], right_on=['Y','Q'])
#depict the limits by fill_between
fig, host = plt.subplots(figsize=(10,5))
plt.plot(data_0050.index[:1270], stock_time['limits'], alpha=0)
plt.fill_between(data_0050.index[:1270], 0, 175, where=stock_time['limits']==1, alpha=0.2, label="prosperity")
plt.fill_between(data_0050.index[:1270], 0, 175, where=stock_time['limits']==2, alpha=0.2, label="slow growth")
plt.fill_between(data_0050.index[:1270], 0, 175, where=stock_time['limits']==3, alpha=0.2, label="recession")
plt.fill_between(data_0050.index[:1270], 0, 175, where=stock_time['limits']==4, alpha=0.2, label="recovery")
host.set_xlabel('Time', fontsize=14)
host.set_ylabel('Price', color='r', fontsize=14)
par1 = host.twinx()
par1.set_ylabel('Price', fontsize=14)
host.plot(data_0050, c='r', label='0050')
par1.plot(data_00679B, c='b', label='00679B')
par1.plot(data_00713, c='y', label='00713')
plt.title('ETF price', fontsize=20)
host.legend(loc='upper left')
plt.legend(loc=(0.01,0.57))
plt.subplots_adjust(right=0.75)
plt.grid(visible=1)
plt.show
#see the correlation between each etfs
cor_matrix_etfs = np.corrcoef(stock.values.T)
plt.figure()
sns.set(font_scale=1.)
hm = sns.heatmap(cor_matrix_etfs, cbar=True, square=True,annot=True,fmt='.2f', annot_kws={'size':15},
                 yticklabels=stock.columns, xticklabels=stock.columns, linewidth=0.5, cmap='crest')
plt.title('correlation of ETFS', fontsize=18)
plt.savefig('correlation_of_ETFS.png', dpi=400)

#calulate the individual performance: risk and return (std and expected return)
returns = stock.pct_change()
returns = returns[1:]
risk = np.sqrt(returns.var() * 252)
cov_matrix = returns.cov()*252
expected_returns = returns.mean()*252
sharpe_ratio = expected_returns/risk
stocks_weights = np.array([1/3,]*total_stocks)
etfs_info = pd.DataFrame([expected_returns, risk, sharpe_ratio], index=['returns', 'risk','sharpe_ratio']).T
#DEPICT into table
plt.figure()
sns.set(font_scale=1.)
hm = sns.heatmap(etfs_info, cbar=False, square=True,annot=True,fmt='.2f', annot_kws={'size':15},
                 yticklabels=etfs_info.index, xticklabels=etfs_info.columns, linewidth=0.5,  cmap='crest')
plt.title('Performance of ETFs', fontsize=20)
plt.savefig('Performance_of_ETFs.png', dpi=400)
#default port folio with same weight
portfolio_return = sum(stocks_weights*expected_returns)
portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, cov_matrix, stocks_weights.T]))
#even weight to each etfs
print('預期報酬率為: ' + str(round(portfolio_return, 4)))
print('風險為: ' + str(round(portfolio_risk, 4)))
# Randomly allocate the ratio of the assets, epochs setting at 10000
risk_list = []
return_list = []
simulations_target = 10**4
for _ in range(simulations_target):

    # random weighted
    weight = np.random.rand(total_stocks)
    weight = weight / sum(weight)

    # calculate result
    ret = sum(expected_returns * weight)
    risk = np.sqrt(reduce(np.dot, [weight, cov_matrix, weight.T]))

    # record
    return_list.append(ret)
    risk_list.append(risk)

fig = plt.figure(figsize=(10, 6))
fig.suptitle('Allocation Simulations', fontsize=18, fontweight='bold')
plt.scatter(risk_list, return_list, color='#2B8AD5', edgecolors='#000000', alpha=0.2)
plt.title(f'n={simulations_target}', fontsize=16)
plt.xlabel('Risk')
plt.ylabel('Return')
plt.show

#find minimal volatity point by scipy.optimize.minimize
def standard_deviation(weights):
    return np.sqrt(reduce(np.dot, [weights, cov_matrix, weights.T]))
# constraints sum(x) = 1, eq means the outcome of the function would be zero
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stocks))
constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
minimize_variance = minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)

mvp_risk = minimize_variance.fun
mvp_return = sum(minimize_variance.x * expected_returns)

print('風險最小化投資組合預期報酬率為:' + str(round(mvp_return, 4)))
print('風險最小化投資組合風險為:' + str(round(mvp_risk, 4)))

for i in range(total_stocks):
    stock_symbol = str(stock.columns[i])
    weighted = str(format(minimize_variance.x[i], '.4f'))
    print(f'{stock_symbol} 佔投資組合權重 : {weighted}')
#find maximal sharpe point by scipy.optimize.minimize with def the reciprocal of sharpe
def reciprocal_sharpe(weights):
    return np.sqrt(reduce(np.dot, [weights, cov_matrix, weights.T]))/sum(weights * expected_returns)

minimize_reciprocal_sharpe = minimize(reciprocal_sharpe, x0=x0, constraints=constraints, bounds=bounds)

best_sharpe_risk = standard_deviation(minimize_reciprocal_sharpe.x)
best_sharpe_return = sum(minimize_reciprocal_sharpe.x * expected_returns)

print('最佳投資組合預期報酬率為:' + str(round(best_sharpe_return, 4)))
print('最佳投資組合風險為:' + str(round(best_sharpe_risk, 4)))

for i in range(total_stocks):
    stock_symbol = str(stock.columns[i])
    weighted = str(format(minimize_reciprocal_sharpe.x[i], '.4f'))
    print(f'{stock_symbol} 佔投資組合權重 : {weighted}')
    
#depict the whole picture
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stocks))

efficient_fronter_return_range = np.arange(0.05, 0.22, .005)
efficient_fronter_risk_list = []

for i in efficient_fronter_return_range:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * expected_returns) - i}]
    efficient_fronter = minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
    efficient_fronter_risk_list.append(efficient_fronter.fun)

risk_free = 0.0172

fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot()

fig.subplots_adjust(top=0.85)
ax0 = ax.scatter(risk_list, return_list,
                 c=(np.array(return_list)-risk_free)/np.array(risk_list),
                 marker='o')
ax.plot(efficient_fronter_risk_list, efficient_fronter_return_range, linewidth=1, 
    color='#251f6b', marker='o', markerfacecolor='#251f6b', markersize=5)
ax.plot(mvp_risk, mvp_return, 'o', color='#4F5FC4', markerfacecolor='#FCFF20',  
    markersize=16, label='mvp')
ax.plot(best_sharpe_risk, best_sharpe_return, '^', color='#4F5FC4', markerfacecolor='#FCFF20',  
    markersize=16, label='best sharpe-ratio')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Efficient Frontier', fontsize=22, fontweight='bold')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
ax.legend(loc='lower right')

fig.colorbar(ax0, ax=ax, label='Sharpe Ratio')
plt.savefig('EF.png', dpi=400)

#select all the stock data in limit2: slow growth, and calulate the weight
returns.index = range(len(returns))
stock_time_minus1 = stock_time.drop(0)
stock_time_minus1.index = stock_time_minus1.index-1
new_returns = pd.concat([returns, stock_time_minus1], axis=1)
masklimit2 = new_returns['limits'] == 2
new_returns_limit2 = new_returns[masklimit2]
#calculate new return and risk and sharpe ratio based on limit2
risk_limit2 = np.sqrt(new_returns_limit2.iloc[:, :3].var() * 252)
cov_matrix_limit2 = new_returns_limit2.iloc[:, :3].cov()*252
expected_returns_limit2 = new_returns_limit2.iloc[:, :3].mean()*252
sharpe_ratio_limit2 = expected_returns_limit2/risk_limit2
stocks_weights = np.array([1/3,]*total_stocks)
etfs_info_limit2 = pd.DataFrame([expected_returns_limit2, risk_limit2, sharpe_ratio_limit2], 
                         index=['returns', 'risk','sharpe_ratio']).T

plt.figure()
sns.set(font_scale=1.)
hm = sns.heatmap(etfs_info_limit2, cbar=False, square=True,annot=True,fmt='.3f', annot_kws={'size':15},
                 yticklabels=etfs_info_limit2.index, xticklabels=etfs_info_limit2.columns, linewidth=0.5,  cmap='crest')
plt.title('Performance of ETFs during limit2', fontsize=16)
plt.savefig('Performance_of_ETFs_limit2.png', dpi=400)

risk_list_limit2 = []
return_list_limit2 = []
simulations_target = 10**4
for _ in range(simulations_target):

    # random weighted
    weight = np.random.rand(total_stocks)
    weight = weight / sum(weight)

    # calculate result
    ret = sum(expected_returns_limit2 * weight)
    risk = np.sqrt(reduce(np.dot, [weight, cov_matrix_limit2, weight.T]))

    # record
    return_list_limit2.append(ret)
    risk_list_limit2.append(risk)

fig = plt.figure(figsize=(10, 6))
fig.suptitle('Allocation Simulations_limit2', fontsize=18, fontweight='bold')
plt.scatter(risk_list_limit2, return_list_limit2, color='#2B8AD5', edgecolors='#000000', alpha=0.2)
plt.title(f'n={simulations_target}', fontsize=16)
plt.xlabel('Risk')
plt.ylabel('Return')
plt.show
# find the mvp and sharpe best point
def standard_deviation_limit2(weights):
    return np.sqrt(reduce(np.dot, [weights, cov_matrix_limit2, weights.T]))
# constraints sum(x) = 1, eq means the outcome of the function would be zero
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stocks))
constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
minimize_variance_limit2 = minimize(standard_deviation_limit2, x0=x0, constraints=constraints, bounds=bounds)

mvp_risk_limit2 = minimize_variance_limit2.fun
mvp_return_limit2 = sum(minimize_variance_limit2.x * expected_returns_limit2)

print('風險最小化投資組合預期報酬率為:' + str(round(mvp_return_limit2, 4)))
print('風險最小化投資組合風險為:' + str(round(mvp_risk_limit2, 4)))

for i in range(total_stocks):
    stock_symbol = str(stock.columns[i])
    weighted = str(format(minimize_variance_limit2.x[i], '.4f'))
    print(f'{stock_symbol} 佔投資組合權重 : {weighted}')
#find maximal sharpe point by scipy.optimize.minimize with def the reciprocal of sharpe
def reciprocal_sharpe_limit2(weights):
    return np.sqrt(reduce(np.dot, [weights, cov_matrix_limit2, weights.T]))/sum(weights * expected_returns_limit2)

minimize_reciprocal_sharpe_limit2 = minimize(reciprocal_sharpe_limit2, x0=x0, constraints=constraints, bounds=bounds)

best_sharpe_risk_limit2 = standard_deviation_limit2(minimize_reciprocal_sharpe_limit2.x)
best_sharpe_return_limit2 = sum(minimize_reciprocal_sharpe_limit2.x * expected_returns_limit2)

print('最佳投資組合預期報酬率為:' + str(round(best_sharpe_return_limit2, 4)))
print('最佳投資組合風險為:' + str(round(best_sharpe_risk_limit2, 4)))

for i in range(total_stocks):
    stock_symbol = str(stock.columns[i])
    weighted = str(format(minimize_reciprocal_sharpe_limit2.x[i], '.4f'))
    print(f'{stock_symbol} 佔投資組合權重 : {weighted}')
    
#depict the whole picture
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stocks))

efficient_fronter_return_range_limit2 = np.arange(0.02, 0.078, .005)
efficient_fronter_risk_list_limit2 = []

for i in efficient_fronter_return_range_limit2:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * expected_returns_limit2) - i}]
    efficient_fronter_limit2 = minimize(standard_deviation_limit2, x0=x0, constraints=constraints, bounds=bounds)
    efficient_fronter_risk_list_limit2.append(efficient_fronter_limit2.fun)

risk_free = 0.0172

fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot()

fig.subplots_adjust(top=0.85)
ax0 = ax.scatter(risk_list_limit2, return_list_limit2,
                 c=(np.array(return_list_limit2)-risk_free)/np.array(risk_list_limit2),
                 marker='o')
ax.plot(efficient_fronter_risk_list_limit2, efficient_fronter_return_range_limit2, linewidth=1, 
    color='#251f6b', marker='o', markerfacecolor='#251f6b', markersize=5)
ax.plot(mvp_risk_limit2, mvp_return_limit2, 'o', color='#4F5FC4', markerfacecolor='#FCFF20',  
    markersize=16, label='mvp')
ax.plot(best_sharpe_risk_limit2, best_sharpe_return_limit2, '^', color='#4F5FC4', markerfacecolor='#FCFF20',  
    markersize=16, label='best sharpe-ratio')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Efficient Frontier_limit2', fontsize=22, fontweight='bold')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
ax.legend(loc='upper right')

fig.colorbar(ax0, ax=ax, label='Sharpe Ratio')
plt.savefig('EF_limit2.png', dpi=400)
