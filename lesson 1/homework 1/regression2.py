import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data = pd.read_csv("GANtest/lesson 1/homework 1/work/hw1_data/train.csv",encoding = 'big5')
print(pd.__version__)

#提取需要的数字部分
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# feature collecting1
# 将原始 4320 * 24 的资料依照每个月份重组成 12 个 18 (特征) * 480 (小时) 的资料。
month_data={}
for month in range(12):
    sample = np.empty([18,480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24]  =   raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data [month] = sample

# feature collecting2
# 每个月会有 24*20 = 480 小时，每 9 小时形成一个数据，每个月会有 471 个数据，
# 总资料数为 471 * 12 笔，而每笔数据 有 18 * 9 的 特征 (一小时 18 个 特征 * 9 小时)。
# 对应的 目标 则有 471 * 12 个(第 10 个小时的 PM2.5)
x = np.empty([12*471,18*9],dtype = float)
y = np.empty([12*471,1],dtype = float)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour,:] = month_data[month][:,day *24 + hour : day * 24 + hour + 9].reshape(1,-1)  #包含所有数据且将 每个小时的18个参数 按顺序都放在一行
            y[month * 471 + day * 24 + hour,0] = month_data[month][9,day * 24 + hour + 9] #只包含PM2.5的值且将 每个小时的pm2.5值 放在一行

print('原始数据为：',x)
print('PM2.5数据为：',y)


# 归一化
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

print('归一化后的变量为：',x)

#分割训练集和测试集
import math
x_train_set = x[: math.floor(len(x) * 1), :]#拿走前0.75的数据，即9个月的数据
y_train_set = y[: math.floor(len(y) * 1), :]
x_validation = x[math.floor(len(x) * 0.75): , :]#拿走后0.25的数据，即3个月的数据
y_validation = y[math.floor(len(y) * 0.75): , :]
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))



#训练
dim = 18 * 9 + 1
w = np.zeros([dim,1])
x_train_set = np.concatenate((np.ones([12 * 471, 1]), x_train_set), axis = 1).astype(float)  #np.ones([12 * 471, 1]), x)是为了增加 w 中的常数项系数
learning_rate = 10
iter_time = 100000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
loss_mat=[]
lt = []
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set , 2))/471/9)#rmse 均方根误差
    if(t%learning_rate==0):
        lt.append(int(t/learning_rate))
        loss_mat.append(float(loss))
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set ) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

plt.plot(lt,loss_mat)
plt.title('loss in the training set')
plt.xlabel('training times / 1000')
plt.ylabel('loss')
plt.show()
np.save('GANtest/lesson 1/homework 1/work/weight.npy', w)
print('权重1为:',w)


#测试模型的准确度
test_data = x_validation
test_data[test_data == 'NR'] = 0
test_x = np.empty([1413, 18*9], dtype = float)
test_x = test_data
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]  #归一化
test_x = np.concatenate((np.ones([1413, 1]), test_x), axis = 1).astype(float)

#预测
w = np.load('GANtest/lesson 1/homework 1/work/weight.npy')
y_ans = np.dot(test_x, w) 
err = y_validation - y_ans
err_all = np.sum(err)
print('平均error为',err_all/np.sum(y_validation))



