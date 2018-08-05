
# coding: utf-8

# In[57]:


import numpy as np
w=0.5
b=0.2
x=np.random.randn(10000,1)
sigx=1/(1+np.exp(-(w*x+b)))
y=np.zeros(sigx.shape)
y[sigx>0.5]=1


# In[65]:


def logreg_test(x,y,alpha,nmax):
    X=np.append(np.ones((x.shape[0],1)),x,axis=1)
    m,n=X.shape
    w=np.zeros((n,1))
    #print((0,w))
    for i in range(nmax):
        A=1/(1+np.exp(-np.dot(X,w)))
        J=np.sum(-(y*np.log(A)+(1-y)*np.log(1-A)))
        dZ=A-y
        dw=np.dot(X.T,dZ)/m
        w-=alpha*dw
        if np.remainder(i,np.floor(nmax/4))==0:
            print((i,w,J))
    return w


# In[59]:


def test_logreg_perf(x,y,alpha,nmax):
    w_est=logreg_test(x,y,alpha,nmax)
    sigx_hat=1/(1+np.exp(-(w_est[0]+w_est[1]*x)))
    y_hat=np.zeros(sigx_hat.shape)
    y_hat[sigx_hat>0.5]=1
    erate=np.absolute(y_hat-y).sum()/x.shape[0]
    print('Error Rate is '+str(erate/100)+'%')


# In[62]:


test_logreg_perf(x,y,0.001,10000)


# In[64]:


test_logreg_perf(x,y,0.005,10000)


# In[66]:


test_logreg_perf(x,y,0.01,1000)


# In[67]:


test_logreg_perf(x,y,0.005,1000)


# In[68]:


test_logreg_perf(x,y,0.05,2000)

