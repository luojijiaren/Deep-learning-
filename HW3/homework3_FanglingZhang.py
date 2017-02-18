# Problem 2
import numpy as np

def J (w, faces, labels, alpha = 0.):
    y_hat=np.dot(faces,w)
    dif=y_hat-labels
    J=1/2*np.dot(dif.T,dif)+alpha/2*np.dot(w.T,w)
    return J  

def gradJ (w, faces, labels, alpha = 0.):
    dif=np.dot(faces,w)-labels
    gradJ=np.dot(faces.T,dif)+alpha*w
    return gradJ 


if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

        A1=np.dot(trainingFaces.T,trainingFaces)
        A=A1+0.000001*np.identity(576)
        eval,evec = np.linalg.eigh(A)
        eval2=np.power(eval,-1/2)
        fai=np.diag(eval2)
        L=np.dot(evec,fai)
        md_trainingFaces=np.dot(trainingFaces,L)
        return md_trainingFaces
        
        
w = np.zeros(md_trainingFaces.shape[1])  # Or set to random vector
J_dif=0.1
Des_gradJ= np.zeros(md_trainingFaces.shape[1]) 
alpha=0
Cost=[]
while J_dif > 0.001:
    J0=J (w, md_trainingFaces, trainingLabels,alpha)
    Des_gradJ=gradJ (w, md_trainingFaces, trainingLabels,alpha)
    w=w-0.1*Des_gradJ
    J1=J (w, md_trainingFaces, trainingLabels,alpha)
    J_dif=J0-J1
    Cost.append(round(J1))
print("Training cost: {}".format(Cost))
        

        
        
        
        
# Problem 3
        
import numpy as np

def J (w, faces, labels, alpha = 0.):
    m=labels.size
    z=np.dot(faces,w)
    y_hat=np.reciprocal(1+np.exp(-z))
    J=-1/m*(np.dot(labels.T,np.log(y_hat))+np.dot((1-labels).T,np.log(1-y_hat)))+alpha/2*np.dot(w.T,w)
    return J  

def gradJ (w, faces, labels, alpha = 0.):
    m=labels.size
    z=np.dot(faces,w)
    y_hat=np.reciprocal(1+np.exp(-z))
    dif=y_hat-labels
    gradJ=1/m*np.dot(faces.T,dif)+alpha*w
    return gradJ 

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")
        
w = np.zeros(trainingFaces.shape[1])  # Or set to random vector
J_dif=0.1
Des_gradJ= np.zeros(trainingFaces.shape[1]) 
alpha=0
Cost=[]
for i in range(15000):
    J0=J (w, trainingFaces, trainingLabels,alpha)
    Des_gradJ=gradJ (w, trainingFaces, trainingLabels,alpha)
    w=w-0.25*Des_gradJ
    J1=J (w, trainingFaces, trainingLabels,alpha)
    J_dif=J0-J1
    Cost.append(J1)
print(w)
print("Training cost: {}".format(Cost[-20:])) 


def method4 (trainingFaces, trainingLabels):
    z=np.log(trainingLabels)-np.log(1-trainingLabels)
    w = np.zeros(trainingFaces.shape[1]) 
    mult_faces=np.dot(trainingFaces.T,trainingFaces)
    mult_labels=np.dot(trainingFaces.T,z)
    w=np.linalg.solve(mult_faces,mult_labels)
    return w


def reportCosts (w, trainingFaces, trainingLabels, alpha = 0.):
    print ("Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha)))

w4 = method4(trainingFaces, trainingLabels)
reportCosts(w4, trainingFaces, trainingLabels) 
    

#check gradient
from scipy.optimize import check_grad
check_grad(J, gradJ,w,trainingFaces, trainingLabels)
        
#compare against sklearn.linear model.LogisticRegression        
from sklearn.linear_model import LogisticRegression
lrc=LogisticRegression(C=1e-2,fit_intercept=False)
lrc.fit(trainingFaces, trainingLabels)
y_hat=lrc.predict_proba(trainingFaces)[:,1]
y=trainingLabels
m=trainingLabels.size     
J=-1/m*(np.dot(y.T,np.log(y_hat))+np.dot((1-y).T,np.log(1-y_hat)))
J
