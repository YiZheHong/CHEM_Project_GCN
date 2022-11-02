import pandas as pd
dic = {}
models = ['GCN']
tags = ['train','valid','test']
for tag in tags:
    with open(f'./2129Data/GCN_fold1_{tag}Loss.txt') as f:
        lines = f.read()
        l = [round(float(num),2) for num in lines[1:-1].split(', ')]
        dic[tag] = l
print(dic)
df2 = pd.DataFrame.from_dict(dic)
df2.to_csv('./loss.csv', encoding='utf-8',index=False)

