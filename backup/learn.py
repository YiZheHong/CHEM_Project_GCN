import numpy as np
import scipy.sparse as sp
from csv import reader
import GN
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
start = time.time()
import pandas as pd
from scipy.stats import iqr

outlier_threshHold = 10000
bar = 1000

sum = 0
min = 0
max =1000000
y = {}
yList = []
node_num= []
node_numD = {2:0,3:0,4:0,5:0,6:0,7:0}

charges = []
chargesD = {0:0,1:0,-1:0}

def load_graph_label(path="./CHEM_Data.csv", seed=5):
    print('loading data from', path)
    data = []
    max_node = 0
    with open(path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)

        # Iterate over each row in the csv using reader object
        j = 0
        for row in csv_reader:
            if j != 0:
                data.append(row)
                if len(row) - 3 > max_node:
                    max_node = len(row) - 3
            j += 1
    random.seed(seed)
    random.shuffle(data)

    l = [GN.Graph(graph, max_node) for graph in data]
    return l
l = load_graph_label(path="data/CHEM_Data.csv", seed=5)

for i in range(0,outlier_threshHold,bar):
    if i + bar == 500:
        y["        "+str((i + bar) // 100) + "   * 10^2"] =0
    else:
        y[str((i+bar)//100)]= 0

def addIn(value):
    for i in range(0, outlier_threshHold, bar):
        if value > i and value < i+bar:
            if i+bar == 500:
                y["        "+str((i + bar) // 100)+"   * 10^2"] += 1
            else:
                y[str((i+bar)//100)] +=1
def addin(value):
    if value == 0:
        chargesD[0]+=1
    else:
        chargesD[1]+=1
def addN(value):
    for i in range(2,8):
        if value == i:
            node_numD[i]+=1




for i in l:
    n = i.num_of_chem
    if n < min:
        min = n
    if n  > max:
        max = n
    sum+=i.num_of_chem
    if i.y_value <=outlier_threshHold and i.y_value>100:
        addIn(i.y_value)
        # yList.append(i.y_value)
        yList.append(i.y_value)
        node_num.append(i.num_of_chem)
        addin(i.charge)
        addN(i.num_of_chem)
        charges.append(int(i.charge))
# print(np.sum(y)/2048,min, max)

# print(iqr(yList)*3)

print(y)
print(np.mean(yList))
print(np.median(yList))
print(np.std(yList))

xLim, xTicks,yLim,yTicks,xlabel,ylabel,title = 1,1,1500,200,"Charges","Frequency","Distribution of charges"
def plotHist(xLim, xTicks,yLim,yTicks,xlabel,ylabel,title,figSize = (10, 10)):
    plt.figure(figsize=(5, 5))
    plt.hist(chargesD,bins = int(2),color='blue', edgecolor='black')
    plt.xlim(0, xLim)
    plt.xticks(np.arange(0, xLim+xTicks, xTicks))
    plt.yticks(np.arange(0, yLim+yTicks, yTicks))
    plt.ylim(0, yLim)
    plt.xlabel(xlabel,size = 20)
    plt.ylabel(ylabel,size = 20)
    plt.title(title,size = 20)
    plt.show()

def q123(y):
    y = removeNone(y)
    # print(y)
    commutes = pd.Series(y)
    print(commutes)
    plt.rcParams['font.size'] = 24

    # commutes.plot.bar(color='#607c8e')
    commutes.plot(figsize=(12, 9),kind = 'bar',color='#607c8e')

    # for p in commutes.patches:
    #     commutes.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    # plt.title('Distribution of nodes')
    plt.xticks(rotation=0)  # Rotates X-Axis Ticks by 45-degrees

    plt.xlabel('Group of Y-values')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
def removeNone(d):
    for key in list(d.keys()):
        if d[key] == 0:
            del d[key]
    return y
q123(y)
# plotHist(xLim, xTicks,yLim,yTicks,xlabel,ylabel,title)
