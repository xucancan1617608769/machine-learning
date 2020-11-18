import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from plot import plot_decision_regions
class perception(object):
    def __init__(self,alpha=0.01,loop=50):  #初始化参数
        self.alpha=alpha
        self.loop=loop

    def train(self,x,y):  #训练模型,x为m×n的矩阵，是多个数据集,y是m×1数据集标签
        self.w=np.zeros(x.shape[1])  #初始化w,b
        self.b=0
        self.erros=[]
        err=0
        for _ in range(self.loop):
            for xi,yi in zip(x,y):  #把它变成[（x1,y1）,(x2,y2),..]的形式
                self.w=self.w+self.alpha*(yi-self.predict(xi))*xi   #用了梯度下降算法
                upd=yi-self.predict(xi)
                self.b=self.b+self.alpha*upd
                err+=int(upd)  #统计更新和，以便知道什么时候w,b趋向于稳定状态
            if(err==0):
                break
            self.erros.append(err)
        return self
                
    def predict(self,x):   #预测值
        t=np.dot(x,self.w)+self.b  #对每一个数据进行w*x+b线性转化
        return np.where(t>0.0,1,-1)  #预测值

def main():
    iris = load_iris()
    X = iris.data[:100, [0, 2]]  #提取100个带有两个特征值的数据集
    y = iris.target[:100]        #提取100个标签
    #print(y)
    y = np.where(y == 1, 1, -1)  #构建正规标签
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)  #用来随机划分样本数据为训练集和测试集的
    
    ppn = perception(alpha=0.1, loop=10)
    ppn.train(X_train, y_train)  #训练集训练模型
    
    plot_decision_regions(ppn, X, y)#打印模型
    
main()
