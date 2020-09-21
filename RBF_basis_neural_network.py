import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import norm,pinv

np.random.seed(20)

class RBF:
    '''
    RBF的一个神经网络案例书写参照
    样本X：100*3，100行3列,100个样本3个特征值
    样本Y：100*1，100行1列
    输入层input_dim = 3
    中间层num_centers = 4
    输出层out__dim = 3
    扩展常数 beta = 8

    '''
    def __init__(self,input_dim,num_centers,out_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.out_dim = out_dim
        self.beta = 8                                           #基函数的扩展常数
        self.centers = [np.random.uniform(-1,1,input_dim) for i in range(num_centers)]#构造中间层的组成形式
        self.W = np.random.random((self.num_centers,self.out_dim))#构造权重矩阵w

    def _basisfunc(self,c,d):
        return np.exp(-self.beta * norm(c-d)**2)#构造基函数，norm是二项式计算

    def _calc_Act(self,X):#定义激活函数

        G = np.zeros((X.shape[0],self.num_centers),dtype=np.float)#变量初始化，构造一个初始的矩阵，行列分别确定。shape()函数可以读取矩阵第一位的长度
        for ci ,c in enumerate(self.centers):
            for xi , x in enumerate(X):
                G [xi ,ci] = self._basisfunc(c,x)
        return G



    def train(self,X,Y):
        '''

        :param X: 100*3
        :param Y: 100*1
        :return:
        '''

        #设计中心点，先选用随机的方法
        rnd_idx = np.random.permutation(X.shape[0])[:self.num_centers]#将X中的元素先打散，然后在从前往后选择中心节点的数量,作为新中心,相当于随机选择4个样本作为中心
        self.centers = [X[i,:] for i in rnd_idx]

        #计算RBF激活函数的值

        G = self._calc_Act(X)

        self.W = np.dot(pinv(G),Y)



    def predict(self,X):

        G = self._calc_Act(X)

        Y = np.dot(G , self.W)

        return Y

#构造数据

n = 100

x = np.linspace(-1,1,n).reshape(n,1)

y = np.sin(3*(x+0.5)**3 - 1)

rbf = RBF(1,10,1)
rbf.train(x,y)
z = rbf.predict(x)


plt.plot(x,y,'k-',label = u'real_data')
plt.plot(x,z,'r-',label = u'predict_data')
plt.xlim(-1.2,1.2)
plt.title(u'Example',fontsize = 20 ,color = 'r')
plt.legend(loc = 'upper left')
plt.show()