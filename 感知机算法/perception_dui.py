from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from plot import plot_decision_regions

class perception(object):
    def __init__(self,eta=0.1,loop=100):
        self.eta=eta
        self.loop=loop

    #构建gram矩阵
    def gram(self,X_train):
        n_sample= X_train.shape[0]   #样本的个数
        self.gramer=np.zeros((n_sample,n_sample))   #初始化gram矩阵
        for m in range(n_sample):
            for n in range(n_sample):
                self.gramer[m][n]=np.dot(X_train[m],X_train[n])  #内积矩阵

    def panduan(self,x,y,i):  #判断是否是误分类点，遍历其中一个样本点看是否是误分类点
        temp=self.b
        n_sample=x.shape[0]
        for m in range(n_sample):  #遍历其中一个样本点，检查是否是误分类点
            temp+=self.alpha[m]*y[m]*self.gramer[i][m]
        return y[i]*temp

    def fit(self,x_train,y_train):  #训练模型
        i=0
        x_sample=x_train.shape[0]
        self.alpha=[0]*x_sample    #初始化alpha
        self.w = np.zeros(x_train.shape[1])  #初始化w
        self.b=0   #初始化b
        self.gram(x_train)   #构建gram矩阵
        while(i<x_sample):  #遍历所有的样本点
            if self.panduan(x_train,y_train,i)<=0:  #判断每个点是否是误分类点
                self.alpha[i]+=self.eta   #如果是，更新alpha[i]
                self.b+=y_train[i]*self.eta  #更新 b
                i=0  #说明还有误分类点，继续检查
            else:
                i+=1
        for j in range(self.loop):   #更新w
            self.w+=self.alpha[j]*x_train[j]*y_train[j]
        return self
    def predict(self,x):   #测试data,预测值target
        t=np.dot(x,self.w)+self.b
        return(np.where(t>0.0,1,-1))

def main():
       load=load_iris()
       x=load.data[:100,:2]
       y=load.target[:100]
       y=np.where(y==1,1,-1)
       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
       clf=perception(eta=0.1,loop=30)
       clf.fit(x_train,y_train)  #训练模型
       plot_decision_regions(clf, x, y)

if __name__ == '__main__':
       main()









