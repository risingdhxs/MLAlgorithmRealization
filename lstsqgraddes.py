def lstsqgraddes(A,b,x0,nmax):
    import numpy as np
    alpha=0.00015
    print('Alpha is ' + str(alpha))
#     Storing historical Error and solutions, up to 3
    E=np.zeros([3,1])
    (n,m)=A.shape
    xhist=np.zeros([3,m])
    
    xhist[2,:]=x0.T
    E[2,0]=0.5*np.sum((b-np.dot(A,x0))**2)
    i=1
    while (i<nmax):
        wdel=-np.dot(A.T,(b-np.dot(A,np.array([xhist[2,:]]).T)))
        xhist[0:2,:]=xhist[1:3,:];E[0:2,0]=E[1:3,0]
        xhist[2,:]=xhist[1,:]-alpha*wdel.T
        E[2,0]=0.5*np.sum((b-np.dot(A,np.array([xhist[2,:]]).T))**2)
#         If the change in solution is smaller than threshold, 1%, assume local optima is reached
        if ( (np.sum((xhist[1,:])**2)>0) and ((np.sum((xhist[2,:]-xhist[1,:])**2)/np.sum((xhist[1,:])**2))<0.00000001)):
            print('Local Minima Reached, i='+str(i))
            i=nmax*2
#         If error is decreasing, assume stepping in the right direction, increase the step size
        elif E[2,0]<E[1,0]:
            alpha=alpha*1.5;i=i+1
#         If error increases, need to decrease step size because it's in the wrong direction
        else:
            alpha=alpha/2;i=i+1
        
        if i==nmax:
            print('Reached Maximum Iteration Number')
    return np.array([xhist[2,:]]).T



import numpy as np
import matplotlib.pyplot as plt
import time

p=400000
x1=np.random.rand(p,1)
x2=np.random.rand(p,1)
e=np.random.rand(p,1)-0.5
y=1.2+2.3*x1+1.75*x2+e
y_clean=1.2+2.3*x1+1.75*x2

A=np.column_stack((np.ones((p,1)),x1,x2))

t0_ls=time.time()
wls,resid,rank,s=np.linalg.lstsq(A,y)
t1_ls=time.time()
t_ls=t1_ls-t0_ls
y_ls=wls[0]+wls[1]*x1+wls[2]*x2

t0_gd=time.time()
wgd=lstsqgraddes(A,y,np.zeros((3,1)),10000)
t1_gd=time.time()
t_gd=t1_gd-t0_gd
y_gd=wgd[0]+wgd[1]*x1+wgd[2]*x2


t0_ne=time.time()
wne=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)
t1_ne=time.time()
t_ne=t1_ne-t0_ne

print(np.column_stack(([1.2,2.3,1.75],wls,wgd,wne)))   
print('linalg Least Square Time is '+str(t_ls))
print('Gradient Descent Time is '+str(t_gd))
print('Normal Equation Time is '+str(t_ne))


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# yls=plt.scatter(y,y_ls)
# ygd=plt.scatter(y,y_gd)
# yclean=plt.scatter(y,y_clean)
# plt.legend((yls,ygd,yclean),('Normal Equation','Gradient Descent','Clean Data'))
# plt.grid()
# 
# plt.show()
    
     
        
    