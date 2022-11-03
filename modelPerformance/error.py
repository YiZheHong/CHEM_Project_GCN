
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd
dic = {}
models = ['GCN']
tags =  ['train','test']
# 'modelPerformance/2129Data/ResultRecord/GCN_32_16_128_fold1_trainResult.csv'
def mse(y_true, y_pred):
    return round((y_pred - y_true).mean(),2)

def rmse(y_true,y_pred):
    return round(mean_squared_error(y_true,y_pred,squared=False),2)
def mae(y_true,y_pred):
    return round(mean_absolute_error(y_true,y_pred),2)

for model in models:
    for tag in tags:
        for i in range(1,2):
            loc = f'./2129Data_MSE/ResultRecord/GCN_32_16_128_fold{i}_{tag}Result.csv'
            df = pd.read_csv(loc)
            dic[model+'_'+tag+'_'+str(i)]=[mse(df['y_true'],df['y_pred']),mae(df['y_true'],df['y_pred']),rmse(df['y_true'],df['y_pred'])]
print(dic)
df2 = pd.DataFrame.from_dict(dic)
df2.to_csv('./l.csv', encoding='utf-8',index=False)

# {'GCN_train_1': [-0.1, 4.37, 6.02], 'GCN_test_1': [1.64, 6.42, 10.48]}