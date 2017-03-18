from numpy import *
import numpy as np 
faces=np.load("mnist_train_images.npy") 
labels=np.load("mnist_train_labels.npy") 
valX=np.load("mnist_validation_images.npy") 
valY=np.load("mnist_validation_labels.npy") 
testX=np.load("mnist_test_images.npy") 
testY=np.load("mnist_test_labels.npy")

def J (W1, W2,b1,b2,faces, labels,alpha):  
    z1=np.dot(faces,W1)+b1
    h1=where(z1<0,0,z1)
    z2=np.dot(h1,W2)+b2
    f=exp(z2)
    y_hat=f/f.sum(axis=1)[:,None]    
    m=labels.shape[0]
    Jsum=0
    for i in range(0,m):
        Jsum=Jsum+np.dot(log(y_hat[i,:]),labels[i,:].T)
    Cost=-1/m*Jsum+alpha/2*(sum(diag(np.dot(W1.T,W1)))+sum(diag(np.dot(W2.T,W2))))
    return Cost        

def gradJ (W1, W2,b1,b2,faces, labels,alpha):
    z1=np.dot(faces,W1)+b1
    h1=where(z1<0,0,z1)
    z2=np.dot(h1,W2)+b2
    f=exp(z2)
    y_hat=f/f.sum(axis=1)[:,None]  
    
    gt=np.dot((y_hat-labels),W2.T)*sign(z1)
    m=labels.shape[0] 
    n=W1.shape[0]
    p=W1.shape[1]
    g=zeros([m,n*p])
    for i in range(0,m):
        for j in range(0,p):
            g[i,n*j:n*(j+1)]=gt[i,j]*faces[i,:]
    g2=np.sum(g,axis=0)   #Sum up all W1 gradient of all samples in given dataset
    grad_W1=zeros([n,p])    
    for j in range(0,p):
        for i in range(0,n):
            grad_W1[i,j]=g2[n*j+i]
    grad_W1=grad_W1+alpha*W1       
    grad_W2=np.dot(h1.T,(y_hat-labels))+alpha*W2 #Sum up all W2 gradient of all samples in given dataset
    
    grad_b1=np.sum(gt,axis=0) #Sum up all b1 gradient of all samples in given dataset
    grad_b2=np.sum(y_hat-labels,axis=0) #Sum up all b2 gradient of all samples in given dataset
    #need to add grad_b1,grad_b2
    return grad_W1,grad_W2 ,grad_b1, grad_b2


def testAccuracy(valX,valY,W1,W2,b1,b2):
    m=valY.shape[0]
    b3=repeat(b1,m).reshape(m,b1.shape[1])
    b4=repeat(b2,m).reshape(m,b2.shape[1])
    z1=np.dot(valX,W1)+b3
    h1=where(z1<0,0,z1)
    z2=np.dot(h1,W2)+b4
    f=exp(z2)
    y_hat=f/f.sum(axis=1)[:,None] 
    alpha=0
    J_test=J (W1, W2,b3,b4,valX,valY,alpha)
    n=0
    for i in range(0,m):
        for j in range(0,10):
            if (y_hat[i,j]==np.sort(y_hat[i,:])[-1]):
                y_hat[i,j]=1
            else:
                y_hat[i,j]=0   

        if (y_hat[i,:]==valY[i,:]).all():
            n=n+1
    a=round(n/m,4)
    return J_test, a

def SGD(faces, labels, num_units,alpha,epochs, mini_batch_size, eta, valX,valY):
        W1=random.uniform(-1/sqrt(784),1/sqrt(784),size=(784,num_units)) 
        W2=random.uniform(-1/sqrt(num_units),1/sqrt(num_units),size=(num_units,10))
        b1=0.01*ones([1, num_units])
        b2=0.01*ones([1, 10])
        for j in range(epochs):            
            training_data=np.hstack([faces,labels])
            random.shuffle(training_data)
            m=labels.shape[0] 
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, m, mini_batch_size)]
            for mini_batch in mini_batches[:-1]:
                X= mini_batch[:,:784]
                Y= mini_batch[:,-10:]
                grad_W1,grad_W2,grad_b1,grad_b2= gradJ (W1,W2,b1,b2,X,Y,alpha)
                W1=W1-eta/mini_batch_size*grad_W1
                W2=W2-eta/mini_batch_size*grad_W2
                b1=b1-eta/mini_batch_size*grad_b1
                b2=b2-eta/mini_batch_size*grad_b2
                J1 = J (W1, W2,b1,b2,X,Y,alpha)                    
        J_test,a=testAccuracy(valX,valY,W1,W2,b1,b2)
        return J_test,a
    
def findBestHyperparameters():
    num_units_best=1
    alpha_best=0.1
    mini_batch_size_best=1
    epochs_best=1
    eta_best=0.1
    a=0
    for i in range(0,11):

      
        try:
            num_units=int(input("num_units="))
        except ValueError:
            print("Not an integer value...please try again")
            num_units=int(input("num_units="))
        eta=float(input("eta="))
        try:
            mini_batch_size=int(input("mini_batch_size="))
        except ValueError:
            print("Not an integer value...please try again")
            mini_batch_size=int(input("mini_batch_size="))
        try:
            epochs=int(input("epochs="))
        except ValueError:
            print("Not an integer value...please try again")
            epochs=int(input("epochs="))
        alpha=float(input("alpha="))

        Ji,ai=SGD(faces, labels, num_units,alpha,epochs, mini_batch_size, eta, valX,valY)
        if ai>a:
            a=ai
            print("The best accuracy on the validation set until now: {};".format(a))
            num_units_best=num_units
            alpha_best=alpha
            mini_batch_size_best=mini_batch_size
            epochs_best=epochs
            eta_best=eta

        else:
            print("The best accuracy do not change this time")
        print("-------------------------------------------------------------")
    print("Finally, The best accuracy on the validation set: {};".format(a))
    print("The best hyper parameters--")
    print("num_units_best: {};".format(num_units_best))
    print("alpha_best: {};".format(alpha_best))
    print("mini_batch_size_best: {};".format(mini_batch_size_best))
    print("epochs_best: {};".format(epochs_best))
    print("eta_best: {};".format(eta_best))
    return num_units_best, alpha_best,mini_batch_size_best, epochs_best,  eta_best

findBestHyperparameters()

J_test,a_test=SGD(faces, labels, num_units_best,alpha_best,epochs_best, mini_batch_size_best, eta_best, testX,testY)
print("The cross-entropy cost on the testing set: {}".format('%.3f' %J_test))
print("The accuracy on the testing set: "+"{0: .2%}".format(a_test))