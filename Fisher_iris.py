
# coding: utf-8

# 
# # The Iris Dataset
# 
# This data sets consists of 3 different types of irises'
# (Setosa, Versicolour, and Virginica) petal and sepal
# length, stored in a 150x4 numpy.ndarray
# 
# The rows being the samples and the columns being:
# Sepal Length, Sepal Width, Petal Length	and Petal Width.
# 
# The below plot uses the first two features.
# See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
# information on this dataset.
# 
# 

# In[1]:


import matplotlib.pyplot as plt#绘图
import numpy as np#矩阵运算
from sklearn.datasets import load_iris #数据集
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def LDA_dimensionality(X,y,k):#数据集，标签，目标维数
    label_ = list(set(y))#标签所有种类
    X_classify={}#存放不同标签的数据集合
    
    for label in label_:
        X1=np.array([X[i] for i in range(len(X)) if y[i]==label])#存放标签相同的数据的下标
        X_classify[label]=X1#将相同标签的数据集合作为一个元素存入矩阵中
    
    mju = np.mean(X,axis=0)#压缩行，只计算列的均值,结果为特征各自的均值
    mju_classify={}
    
    for label in label_:
        mju1 = np.mean(X_classify[label],axis=0)#计算每一类的均值
        mju_classify[label] = mju1
    #上面分好类后   开始计算总类内、类间散度矩阵
    #总类内散度矩阵
    Sw=np.zeros((len(mju),len(mju)))
    for i in label_:
        Sw += np.dot((X_classify[i]- mju_classify[i]).T,(X_classify[i]- mju_classify[i]))#>=3类别的总类内散度矩阵
    #类间散度矩阵
    Sb=np.zeros((len(mju),len(mju)))
    for i in label_:
        Sb += len(X_classify[i])*np.dot((mju_classify[i]-mju).T,mju_classify[i]-mju)
        
    #计算Sw-1*Sb的特征值和特征矩阵,特征值的特征向量下标对应特征矩阵每一行
    eig_vals,eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    sorted_indices = np.argsort(eig_vals)#返回对数据进行排序的索引
    topk_eig_vecs = eig_vecs[:-k-1:-1]#提取特征值大的前k个特征向量,作为W*
    return topk_eig_vecs

if '__main__' == __name__:
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    W=LDA_dimensionality(X,y,2)
    X_new=np.dot(X,W.T)
    plt.figure(1)
    plt.scatter(X_new[:,0],X_new[:,1],marker="o",c=y)

    plt.show()


# In[ ]:




