from numpy import *
import numpy as np
trainingX=np.load("mnist_train_images.npy")
trainingY=np.load("mnist_train_labels.npy")
testingX=np.load("mnist_test_images.npy")
testingY=np.load("mnist_test_labels.npy")

def J (w, faces, labels):
    f=exp(np.dot(faces,w))
    y_hat=f/f.sum(axis=1)[:,None]
    m=labels.shape[0]
    Jsum=0
    for i in range(0,m):
        Jsum=Jsum+np.dot(labels[i,:].T,log(y_hat[i,:]))
    J=-1/m*Jsum
    return J  

def gradJ (w, faces, labels):
    f=exp(np.dot(faces,w))
    y_hat=f/f.sum(axis=1)[:,None]
    gradJ=np.dot(faces.T,(y_hat-labels))
    return gradJ 

w=np.zeros((784,10)) # Or set to random vector
J_dif=1
cost=[]
n=0
while J_dif > 0.001:
    J0=J (w, trainingX, trainingY)
    Des_gradJ=gradJ (w, trainingX, trainingY)
    w=w-0.000002*Des_gradJ
    J1=J (w, trainingX, trainingY)
    J_dif=J0-J1
    cost.append(J1)
cost
w

myList=cost[-20:]
b=['%.4f' %elem for elem in myList]
print("Training cost:{} ".format(b) )


        
f=exp(np.dot(testingX,w))
y_hat=f/f.sum(axis=1)[:,None]
m=testingY.shape[0]
n=0
for i in range(0,m):
    for j in range(0,10):
        if (y_hat[i,j]==np.sort(y_hat[i,:])[-1]):
            y_hat[i,j]=1
        else:
            y_hat[i,j]=0   
                                  
    if (y_hat[i,:]==testingY[i,:]).all():
        n=n+1
a=round(n/m,4)
print("The accuracy on the testing set: {}".format(a))