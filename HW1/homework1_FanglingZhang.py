import numpy as np

def problem1 (A, B):
	return A + B

def problem2 (A, B, C):
	return np.dot(A,B)-C

def problem3 (A, B, C):
	return A*B+C.T

def problem4 (x, y):
	return np.dot(x.T,y)

def problem5 (A):
	return np.zeros(A.shape)

def problem6 (A):
	return np.ones(A.shape)

def problem7 (A):
	return np.linalg.inv(A)

def problem8 (A, x):
	return np.linalg.solve(A,x)

def problem9 (A, x):
	return (np.linalg.solve(A.T,x.T)).T

def problem10 (A, alpha):
	return A+alpha*np.eye(A.shape[0])

def problem11 (A, i, j):
	return A[i-1,j-1]

def problem12 (A, i):
	return np.sum(A[i-1,:])

def problem13 (A, c, d):
	return np.mean(A[np.nonzero((A>=c)&(A<=d))])

def problem14 (A, k):
	eval,evec = np.linalg.eig(A)
	return evec[np.argsort(eval)[:k]]

def problem15 (x, k, m, s):
	A=x+m*np.ones(x.shape)
	B=s*np.eye(x.shape[1])
	return np.random.multivariate_normal(A,B)
    
