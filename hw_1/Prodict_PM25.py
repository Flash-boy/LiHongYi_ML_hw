import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 获取原始数据
data = pd.read_csv("./data/train.csv",encoding='big5')
data = data.iloc[:,3:]
data[data=='NR']=0

# raw_data为4320X24
raw_data = data.to_numpy()

# 按月份获取数据
month_data={}
for month in range(12):
    sample = np.empty([18,480])
    for day in range(20):
        sample[:,day*24:(day+1)*24] = raw_data[18*(20*month+day):18*(20*month+day+1),:]
    month_data[month]= sample
    
# 将数据转换为所需格式，x每一行代表一个样本，y每一行为对应的标签
x = np.empty([471*12,18*9],dtype=float)
y = np.empty([471*12,1],dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour >14:
                continue
            x[month*471+day*24+hour,:] = month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1)
            y[month*471+day*24+hour,0] = month_data[month][9,day*24+hour+9]

# 对数据预处理归一化
mean_x = np.mean(x,axis=0)
std_x = np.std(x,axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j]!=0:
            x[i][j]=(x[i][j]-mean_x[j])/std_x[j]


# 使用GD，并且采用线性回归模型
def GD(X,Y,W,learning_rate,Iteration,lambdaL2):
    listCost = []
    for itera in range(Iteration):
        arrayYhat = X.dot(W)
        arrayLoss = arrayYhat-Y
        arrayCost = (np.sum(arrayLoss**2)/X.shape[0])
        listCost.append(arrayCost)

        # 计算梯度,使用L2正则化
        arrayGradient = np.dot(np.transpose(X),arrayLoss) / X.shape[0] + (lambdaL2*W)
        W -= learning_rate*arrayGradient
        if itera%1000==0:
            print("iteration:{}, cost:{}".format(itera,arrayCost))
    return W,listCost

# 使用Adagrad eps项防止分母为0
def Adagrad(X,Y,W,learning_rate,Iteration,lambdaL2):
    eps = 0.000000001
    listCost = []
    arrayGradientSum = np.zeros([X.shape[1],1])
    for itera in range(Iteration):
        arrayYhat = X.dot(W)
        arrayLoss = arrayYhat-Y
        arrayCost = (np.sum(arrayLoss**2)/X.shape[0])
        listCost.append(arrayCost)

        # 计算并且存储以前的梯度
        arrayGradient = np.dot(np.transpose(X),arrayLoss) / X.shape[0] + (lambdaL2*W)
        arrayGradientSum += arrayGradient**2
        arraySigma = np.sqrt(arrayGradientSum+eps)
        W -= learning_rate*arrayGradient/arraySigma

        if itera%1000==0:
            print("iteration:{}, cost:{}".format(itera,arrayCost))
    return W, listCost

###---train---###

# # 由于常数项存在增加一维为1，方便计算长度项梯度
x = np.concatenate((np.ones([x.shape[0],1]),x),axis=1).astype(float)

# GD
intLearningRate = 1e-4
# 正则项为0
arrayW = np.zeros([x.shape[1],1])
arrayW_gd, listCost_gd = GD(X=x,Y=y,W=arrayW,learning_rate=intLearningRate,Iteration=20000,lambdaL2=0)
# 正则项为0.2
arrayW = np.zeros([x.shape[1],1])
arrayW_gd_1, listCost_gd_1 = GD(X=x,Y=y,W=arrayW,learning_rate=intLearningRate,Iteration=20000,lambdaL2=0.2)

# Adagrad
intLearningRate = 1e-1
arrayW = np.zeros([x.shape[1],1])
arrayW_ada, listCost_ada = Adagrad(X=x,Y=y,W=arrayW,learning_rate=intLearningRate,Iteration=20000,lambdaL2=0)

# 可视化每一步的costfunction
itera_x = range(20000)
plt.plot(itera_x,listCost_gd,color='r')
plt.plot(itera_x,listCost_gd_1,color='g')
plt.plot(itera_x,listCost_ada,color='b')
plt.legend(['gd','gd+lambdaL2','ada'])
plt.show()

###---test---###
# 获取测试数据也做相同处理
testdata = pd.read_csv("./data/test.csv",header=None,encoding='big5')
testdata = testdata.iloc[:,2:]
testdata[testdata == 'NR'] = 0
test_data = testdata.to_numpy()
test_x = np.empty([240,18*9],dtype=float)
for i in range(240):
    test_x[i,:]=test_data[18*i:18*(i+1),:].reshape(1,-1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] !=0:
            #测试数据也要用训练数据的均值和方差作归一化处理
            test_x[i][j]=(test_x[i][j]-mean_x[j]) / std_x[j] 
test_x = np.concatenate((np.ones([240,1]),test_x),axis=1).astype(float)


# 获取test数据的正确PM2.5 方便比较test误差 数据为240*1
testdata = pd.read_csv("./data/ans.csv",encoding='big5')
test_data = testdata.iloc[:,1:]
test_data = test_data.to_numpy()
test_y = test_data


# test 数据得到的预测值
pre_y_gd = np.dot(test_x,arrayW_gd)
pre_y_gd_1 = np.dot(test_x,arrayW_gd_1)
pre_y_ada = np.dot(test_x,arrayW_ada)

### 画图预测值和正确值的图
gd_list = pre_y_gd.reshape(240)
gd_1_list = pre_y_gd_1.reshape(240)
ada_list = pre_y_ada.reshape(240)

correct_list = test_y.reshape(240)
test_data_index = range(240)
#正确PM2.5图
plt.figure(figsize=(8,8))
plt.subplot(221)
plt.plot(test_data_index,correct_list,'k--')
plt.title("ANS")
plt.xlabel("test data index")
plt.ylabel("predict result")
# 预测PM2.5图
plt.subplot(222)
plt.plot(test_data_index,gd_list,'b--')
plt.title("GD")
plt.xlabel("test data index")
plt.ylabel("predict result")
plt.subplot(223)
plt.plot(test_data_index,gd_1_list,'g--')
plt.title("GD+lambdaL2")
plt.xlabel("test data index")
plt.ylabel("predict result")
plt.subplot(224)
plt.plot(test_data_index,ada_list,'r--')
plt.title("ada")
plt.xlabel("test data index")
plt.ylabel("predict result")

plt.show()















