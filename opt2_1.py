# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def jacobian(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])

def hessian(x):
    return np.array([[-400*(x[1]-3*x[0]**2)+2,-400*x[0]],[-400*x[0],200]])

def newton(x0):

    print('初始点为:')
    print(x0,'\n')
    W=np.zeros((2,10**3))
    i = 1
    imax = 1000
    W[:,0] = x0 
    x = x0
    delta = 1
    alpha = 1

    while i<imax and delta>10**(-5):
        p = -np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        x0 = x
        x = x + alpha*p
        W[:,i] = x
        delta = sum((x-x0)**2)
        print('第',i,'次迭代结果:', x, "步长： ", alpha)
        # print(x,'\n')
        i=i+1
    W=W[:,0:i]  # 记录迭代点
    return W

if __name__ == "__main__":

    X1=np.arange(-1.5,1.5+0.05,0.05)
    X2=np.arange(-3.5,2+0.05,0.05)
    [x1,x2]=np.meshgrid(X1,X2)
    f=100*(x2-x1**2)**2+(1-x1)**2; 
    plt.contour(x1,x2,f,20) 

    # x0 = np.array([1.2, 1.2])
    x0 = np.array([-1.2,1])
    W=newton(x0)

    plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 
    plt.show()
