import numpy as np
import csv
import math
from scipy import special as ss

def update_sigma(Xmat,newA,newB,newE,newF):
    inv = np.diag((newA/newB).flatten())+(float(newE)/newF)*np.dot(Xmat.T,Xmat)
    cov = np.linalg.inv(inv)
    return cov

def update_mu(X,y,e,f,sigma):
    (N,D) = X.shape
    mu_lambda = float(e)/f
    col_sum = np.sum(X*y,0).reshape(D,1)
    col = mu_lambda*col_sum
    ans = np.dot(sigma,col)
    return ans

#  mu has to be a (d*1) vector
def update_ef(X,y,mu,e,f,newSigma):
    e0 = 1
    f0 = 1
    N = X.shape[0]
    newE = e0 + (N/2.0)
    newF = np.sum(np.square(y-np.dot(X,mu)))+np.trace(np.dot(newSigma,np.dot(X.T,X)))
    newF = f0 + (0.5)*newF
    return newE,newF

def update_ab(a,b,mu,sigma):
    # a,b (101*1)
    a0 = 1e-16
    b0 = 1e-16
    D = mu.shape[0]
    newA = a0 + 0.5 
    diagnal = np.diag(sigma).reshape(D,1)
    tmp = np.square(mu)+ diagnal
    newB = b0 + (0.5)*tmp
    return newA,newB

def compute_logy(X,y,mu,newSigma,e,f):
    (N,D) = X.shape
    first = (N/2.0)*(ss.digamma(e)-np.log(f)-np.log(2*math.pi))
    second1 = np.sum(np.square(y-np.dot(X,mu)))
    second2 = np.trace(newSigma.dot(np.dot(X.T,X)))
    second = second1+second2
    second = second*(0.5)*(float(e)/f)
    ans = first - second
    return ans

def compute_logw(X,y,a,b,mu,newSigma):
    (N,D) = X.shape
    const = -(D/2.0)*np.log(2*math.pi)
    first = (0.5)*np.sum(ss.digamma(a)-np.log(b))
    diagmat = np.diag((a/b).flatten())
    second = np.trace(diagmat.dot(np.dot(mu,mu.T)+newSigma))
    ans = const+first-0.5*second
    return ans

def compute_logAll(a,b):
    a0 = np.ones((D,1))*(1e-16)
    b0 = np.ones((D,1))*(1e-16)
    Expect = (ss.digamma(a)-np.log(b)).reshape(D,1)
    first = np.sum(a0*np.log(b0))+ np.sum((a0-1)*(Expect))
    #first = np.sum((a0-1)*(Expect))
    second = np.sum(b0*((a/b).reshape(D,1)))+ np.sum(ss.gammaln(a0))
    ans = first-second
    return ans

def compute_loglam(e,f):
    ##since it is p(lambda) e0 and f0 is not change !!!
    e0 = 1
    f0 = 1
    ans = e0*np.log(f0)+(e0-1)*(ss.digamma(e)-np.log(f))-f0*(float(e)/f) - ss.gammaln(e0)
    return ans

def compute_logw_post(sigma):
    D = sigma.shape[0]
    deter = 2*sum([np.log(elements) for elements in np.diag(np.linalg.inv(np.linalg.cholesky(sigma)))])
    # questions here, show TA the derivation, E(exp{}) = -(1/2)*trace(I)
    ans = -(0.5)*D*(np.log(2*math.pi)+1)-0.5*deter
    return ans


def compute_logAll_post(a,b):
    ans = np.sum((a-1)*ss.digamma(a)+np.log(b)-a-ss.gammaln(a))
    return ans

def compute_loglam_post(e,f):
    ans = (e-1)*ss.digamma(e)+np.log(f)-e-ss.gammaln(e)
    return ans


# Three different Data Sets

#set 1
reader_X1 = csv.reader(open("X_set1.csv","rb"))
reader_X1 = list(reader_X1)
Xset1 = np.asarray(reader_X1,dtype="float64")

reader_y1 = csv.reader(open("y_set1.csv","rb"))
reader_y1 = list(reader_y1)
yset1 = np.asarray(reader_y1,dtype="float64")

(N,D) = Xset1.shape
Xset1 = Xset1.reshape(N,D)
yset1 = yset1.reshape(N,1)


#set 2
reader_X1 = csv.reader(open("X_set2.csv","rb"))
reader_X1 = list(reader_X1)
Xset1 = np.asarray(reader_X1,dtype="float64")

reader_y1 = csv.reader(open("y_set2.csv","rb"))
reader_y1 = list(reader_y1)
yset1 = np.asarray(reader_y1,dtype="float64")

(N,D) = Xset1.shape
Xset1 = Xset1.reshape(N,D)
yset1 = yset1.reshape(N,1)


#set 3
reader_X1 = csv.reader(open("X_set3.csv","rb"))
reader_X1 = list(reader_X1)
Xset1 = np.asarray(reader_X1,dtype="float64")

reader_y1 = csv.reader(open("y_set3.csv","rb"))
reader_y1 = list(reader_y1)
yset1 = np.asarray(reader_y1,dtype="float64")

(N,D) = Xset1.shape
Xset1 = Xset1.reshape(N,D)
yset1 = yset1.reshape(N,1)


X = Xset1
y = yset1
# mu = ini_mu
# sigma = ini_sigma
mu = np.zeros((D,1))
sigma_diag = np.random.rand(D,1).flatten()
sigma = np.diag(sigma_diag)
e = 1
f = 1
a = np.ones((D,1))*(1e-16)
b = np.ones((D,1))*(1e-16)

newthresh = []
newthresh2 = []
newthresh3 = []
for i in range(500):   
    (a,b) = update_ab(a,b,mu,sigma)
    (e,f) = update_ef(X,y,mu,e,f,sigma)
    sigma = update_sigma(X,a,b,e,f)
    mu = update_mu(X,y,e,f,sigma)
    #A = compute_logAll(a,b) 
    A = compute_loglam(e,f)+compute_logw(X,y,a,b,mu,sigma)+compute_logy(X,y,mu,sigma,e,f)+compute_logAll(a,b) 
    #A = compute_logy(X,y,mu,sigma,e,f)
    A = A -compute_logw_post(sigma) -compute_loglam_post(e,f) -compute_logAll_post(a,b)
    newthresh.append(A)
%matplotlib inline
from matplotlib import pyplot as plt
index = range(500)
plt.plot(index,newthresh)
# plt.plot(index,newthresh3)
# plt.axis([0,500,-2,40])
plt.show()

## Stem plot
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1,D,D)
function = b/a
markerline, stemlines, baseline = plt.stem(x, function, '-.')
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(baseline, 'color', 'r', 'linewidth', 2)

plt.show()

#print the value of 1/E(lambda)
inv_Elambda = f/e
print inv_Elambda

## Question 4 change three z_set here
reader_Z1 = csv.reader(open("z_set1.csv","rb"))
reader_Z1 = list(reader_Z1)
Zset1 = np.asarray(reader_Z1,dtype="float64")
E_w = mu
y_hat = np.dot(X,E_w)


# d
plt.plot(Zset1,y_hat,c='blue')
plt.scatter(Zset1,yset1,c='yellow',s=10)
plt.plot(Zset1,10*np.sinc(Zset1),c='green')












