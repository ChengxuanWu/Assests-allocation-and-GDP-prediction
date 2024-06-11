from self_func import Datasets, get_engine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  cross_val_score, learning_curve, StratifiedKFold
import pandas as pd
import numpy as np

#Loading all the potential data through api, then transfer to Dataframe
#Class Datasets was defined above, for translate the data from 5 layers of json
GDP = Datasets('a018101010/1+3...Q.&.&startTime=2000-Q0&endTime=2024-Q1')
data_GDP = GDP.get_data()
time_GDP = GDP.get_time()

#load this data by exception, due to the structure is different from others
Chain_GDP = Datasets('a018103020/1+2+3.25..Q.&startTime=2000-Q0&endTime=2024-Q1')
data_Chain_GDP = Chain_GDP.get_data()
time_Chain_GDP = Chain_GDP.get_time()

exportation = Datasets('a081201010/1.1..M.&startTime=2001-M1&endTime=2024-M3')
data_exportation = exportation.get_data()
time_exportation = exportation.get_time()

importation = Datasets('a081201010/1.2..M.&startTime=2001-M1&endTime=2024-M3')
data_importation = importation.get_data()
time_importation = importation.get_time()

currency = Datasets('a090501010/2.13+14+15..M.&startTime=2001-M1&endTime=2024-M3')
data_currency = currency.get_data()
time_currency = currency.get_time()

stock = Datasets('a110404010/1...M.&startTime=2001-M1&endTime=2024-M3')
data_stock = stock.get_data()
time_stock = stock.get_time()

salary = Datasets('a046301010/1.1..M.&startTime=2001&endTime=2024-M3')
data_salary = salary.get_data()
time_salary = salary.get_time()

worktime = Datasets('a046401010/1.1..M.&startTime=2001-M1&endTime=2024-M3')
data_worktime = worktime.get_data()
time_worktime = worktime.get_time()

workforce = Datasets('a046101010/1.1..M.&startTime=2001-M1&endTime=2024-M3')
data_workforce = workforce.get_data()
time_workforce = workforce.get_time()

preliminary = Datasets('a120101010/2...M.&startTime=2001-M1&endTime=2024-M3')
data_preliminary = preliminary.get_data()
time_preliminary = preliminary.get_time()

industrial_index = Datasets('a050104010/1...M.&startTime=2001-M1&endTime=2024-M3') 
data_industrial_index = industrial_index.get_data()
time_industrial_index = industrial_index.get_time()

tourism = Datasets('a070107010/1...M.&startTime=2001-M1&endTime=2024-M3')
data_tourism = tourism.get_data()
time_tourism = tourism.get_time()

order = Datasets('a050105010/1.1..M.&startTime=2001&endTime=2024-M3')
data_order = order.get_data()
time_order = order.get_time()

cpi = Datasets('a030101015/1...M.&startTime=2001-M1&endTime=2024-M3')
data_cpi = cpi.get_data()

#After review some papers and observe the parameters adapted by official units, choose this 14 params for prediction
#This just the rough option. it will be choosen by some ways later
cols = ['exportation', 'importation', 'M1A', 'M1B', 'M2','stock', 'salary', 'worktime', 'workforce',
        'preliminary', 'industrial_index', 'tourism', 'order', 'cpi']
predictors = pd.concat([data_exportation, data_importation, data_currency, data_stock, 
                        data_salary, data_worktime, data_workforce, data_preliminary, data_industrial_index,
                        data_tourism, data_order, data_cpi], axis=1)
predictors.columns = cols
predictors.index = time_order['year']+' '+time_order['mon']
#export data
engine = get_engine()
predictors.to_sql('predictors', engine, if_exists='replace',index=False)
#Due to the GDP growth rate had already been calculate with other parameter, so i try to choose 2 response var
#one is definately 1) GDP Growth rate, another 2) real GDP after adjust
target = pd.concat([data_GDP.iloc[:,1], data_Chain_GDP.iloc[:,1]], axis=1)
target.to_sql('targets', engine, if_exists='replace')
#depict the pairplot to see the rough relationship between each data
#find the data M1B M1A M2 are highly related, so i try to get rid of some params, just use m1b to build model
sns.pairplot(predictors, height=2)
plt.tight_layout()
plt.show
#Delete the column M1A and M2
predictors = predictors.drop(['M1A', 'M2'], axis=1)
cols_new = ['exportation', 'importation', 'M1B','stock', 'salary', 'worktime', 'workforce',
        'preliminary', 'industrial_index', 'tourism', 'order', 'cpi']
#add response variable to try removing some lowly related params
#but the scale between two response or between response and explainatory are far away to each other
#So, standardize first. The method i use is to Z-standardization by standardscaler
#And due to choose the predictors to forecast the response for next quarter, so i have to move the index
#And the number of explainatory are three times as response, so i do average the month data to quarter 
#Due to 2024q1 predictors data is aiming to predict 2024q2 data, so explainatory will be 92 rows
lst = []
for i in range(2, len(predictors), 3):
    lst.append((predictors.iloc[i]+predictors.iloc[i-1]+predictors.iloc[i-2])/3)
X = np.array(lst)
X = pd.DataFrame(X[:92], columns=cols_new)
y = target[5:]
std_X = StandardScaler()
std_y = StandardScaler()
# y = zy.fit_transform(y[:, np.newaxis]).flatten(), have to be 2D array, if just 1d, could add axis
X_std = pd.DataFrame(std_X.fit_transform(X), columns=cols_new)
y_std = pd.DataFrame(std_y.fit_transform(y), columns=['Growth_rate', 'Real_GDP'])
Xy_std = pd.concat([X_std, y_std], axis=1)
cols_new = ['exportation', 'importation', 'M1B','stock', 'salary', 'worktime', 'workforce',
        'preliminary', 'industrial_index', 'tourism', 'order', 'cpi', 'Growth_rate', 'Real_GDP']
#join the X and y, then depict the pairplot again, try best to let the model less complicated from over-fitting
#from the plot, i think all the data is more preictable for real GDP due to the high relationship
#try to use value to explain this more objectively
sns.pairplot(Xy_std, height=2)
plt.tight_layout()
plt.show
#pearson's product-moment correlation coefficient ranging between -1 to 1
#require updating the version to 13.0, or the annot would be so wierd
plt.figure(figsize=(8,8))
cor_matrix = np.corrcoef(Xy_std.values.T)
sns.set(font_scale=0.8)
hm = sns.heatmap(cor_matrix, cbar=True, square=True,annot=True,fmt='.2f', annot_kws={'size':6.5},
                 yticklabels=cols_new, xticklabels=cols_new)
plt.title('correlation of varaibles', fontsize=20)
plt.savefig('correlation.png', dpi=400)
#Try to predict Real GDP and rate respectively, and get rid of the corref lower than 0.5
X = X.drop(['preliminary', 'tourism'], axis=1)
#Still have ten explainatory, so maybe use lasso for extract the features assisting to do penalty
#limiting for data size, so i use test_size = 0.2 to run lasso
Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size=0.2, random_state=10)
std_Xtrain = StandardScaler()
std_ytrain = StandardScaler()
Xtrain_std = std_Xtrain.fit_transform(Xtrain)
ytrain_std = std_ytrain.fit_transform(ytrain)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'gray', 'indigo', 'orange', 'lightblue']


coef_rate, coef_real, params = [], [], []
for i in np.arange(-6, 5):
    lasso = Lasso(random_state=10, alpha=10.**i)
    lasso.fit(Xtrain_std, ytrain_std)
    coef_rate.append(lasso.coef_[0])
    coef_real.append(lasso.coef_[1])
    params.append(10.**i)
coef_rate = np.array(coef_rate)
coef_real = np.array(coef_real)
for coef, title in zip(list([coef_rate, coef_real]), ['Growth_rate', 'Real_GDP']):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title(title, fontsize=20)
    for column, color in zip(range(coef.shape[1]), colors):
        plt.plot(params, coef[:, column], label=X.columns[column], color=color)
    plt.axhline(0, color='#000000', linestyle='--', linewidth=2)
    plt.xlim([10**-5, 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('alpha')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', bbox_to_anchor=(1.18, 1.03), ncol=1, fancybox=True)
    plt.show()
#after this procession, the two model have 2 and 4 params for prediciton respectively
#model for predicting for gdprate: cpi and stock
#for real gdp is industrial_index, cpi, m1b, workforce
#then due to we have few data sizes, so i have to use some specific method to get my accuracy higher
#for instance, k-fold validation
#then due to the data would be different everytime, so we have to do standardize every time
#try to build up a pipeline to make things easier
pipe_lasso = make_pipeline(StandardScaler(), Lasso(random_state=10, alpha=0.1))
scores = cross_val_score(estimator=pipe_lasso, X=Xtrain, y=ytrain.iloc[:, 1], cv=10, n_jobs=1)
print('accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# scores = cross_val_score(estimator=pipe_lasso, X=Xtrain, y=ytrain.iloc[:, 1], cv=10, n_jobs=-1, 
#                          scoring='neg_mean_squared_error')
# print('mse: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
scores = cross_val_score(estimator=pipe_lasso, X=Xtrain, y=ytrain.iloc[:, 0], cv=10, n_jobs=1)
print('accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# the accuracy for GDP GROWTH are poor, so just predict the GDP_real then calculate the growth rate
ytrain = ytrain.iloc[:,1]
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lasso, X=Xtrain, y=ytrain,
                                                        train_sizes=np.linspace(0.1, 1., 10), cv=10, 
                                                        n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='b', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='b')

plt.plot(train_sizes, test_mean, color='g', marker='^', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')

plt.grid(visible=True)
plt.xlabel('The number of training samples')
plt.ylabel('Accuracy')
plt.title('Learning curve', fontsize=15)
plt.legend(loc='lower right')
plt.ylim([0.4, 1.02])
plt.show

pipe_lasso.fit(Xtrain, ytrain)
y_pred = pipe_lasso.predict(Xtest)
print(pipe_lasso.score(Xtest, ytest.iloc[:,1]))
print(mse(ytest.iloc[:,1], y_pred))

plt.plot(range(1, len(y_pred)+1), y_pred, color='b')
plt.scatter(range(1, len(y_pred)+1), ytest.iloc[:,1], edgecolors='y')
plt.show
#predict the next quarter's economy status
cols_newx = ['exportation', 'importation', 'M1B','stock', 'salary', 'worktime', 'workforce',
        'preliminary', 'industrial_index', 'tourism', 'order', 'cpi']
new_x = pd.DataFrame(np.array(lst[-1])).T
new_x.columns = cols_newx
new_x.drop(['preliminary', 'tourism'], axis=1, inplace=True)
new_y_pred = pipe_lasso.predict(new_x)
new_growth_rate = (new_y_pred[0]-y.iloc[-4,1])/y.iloc[-4,1]*100
new_momentum = (new_growth_rate-y.iloc[-1,0])
print('Real GDP after chain adjustment of 2024 Q2:', new_y_pred[0])
print('Predicted growth rate of 2024 Q2:', new_growth_rate)
print('Predict momentum: ', new_momentum)
print('The predicted limit is 2, which means slow growth!')
