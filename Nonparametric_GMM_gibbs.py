import numpy as np
import csv
from pylab import *
import scipy
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.special import digamma
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.special import gammaln

## for i in range(N) we need to run this, since "nj-i need to be update after each ci is sampled"
def compute_assign(c_i,idx):
    N = c_i.shape[0]
    cluster_list = []
    for cluster in range(N):
        if cluster != c_i[idx]:
            cluster_list.append(len(np.where(c_i==cluster)[0]))
        else:
            cluster_list.append(len(np.where(c_i==cluster)[0])-1)
    return np.asarray(cluster_list)

def update_phi(X, a, B, c, m):
    s, d = X.shape
    xbar = np.mean(X,0)
    X_tmp = X-xbar
    sum_mat = np.dot(X_tmp.T,X_tmp)
    xmxbar = (xbar-m).dot((xbar-m).T)
    mj = (c/(s+c))*m+ (1/(s+c))*np.sum(X,axis=0)
    cj = s + c 
    aj = s + a
    Bj = B + sum_mat + (s/(a*s+1.0))*xmxbar
    precision = stats.wishart.rvs(aj,inv(Bj))
    mu = stats.multivariate_normal.rvs(mean=mj,cov=np.linalg.inv(cj*precision))
    return mu, precision

def compute_px(x, a, B, c, m):
    d = 2
    x_mean = m.reshape(1,d)
    x_center = (x-x_mean).reshape(d,1)
    const = (c/(pi*(1.0+c)) )**(d/2)  
    middle = np.linalg.det(B+ (c/(1.0+c))*np.dot(x_center,x_center.T) )**(-(a+1.0)/2.0) / (det(B)**(-a/2.0))
    tmp = 0
    for j in range(int(d)):
        tmp = tmp + np.sum(gammaln((a+1.0)/2.0- float(j)/2.0)-gammaln(a/2.0-float(j)/2.0))
    log_gamma = np.exp(tmp)
    ans = const*middle*log_gamma
    return ans


def Gibbs_GMM(X,iters):
    n,d = X.shape
    c = 0.1
    a = float(d)
    m = np.mean(X,axis=0)
    X_new = X-m
    B = c*float(d)*np.dot(X_new.T,X_new)/float(n) #emp_cov
    mu = [np.zeros(n) for i in range(n)]
    precision = [np.zeros((n,n)) for i in range(n)]
    alpha =1.0
    c_i = np.zeros(n)
    phi = np.zeros((n,n))
    phi[:,0]=1.0
    mu[0], precision[0] = update_phi(X,a,B,c,m)
    num_clusters = []
    for t in range(iters):
        phi = np.zeros((n,n))
        for i, xi in enumerate(X.values):
            ntmp = compute_assign(c_i,i)
            n_j = np.where(ntmp>0)[0]
            for j in n_j:
                phi[i,j] = mvn.pdf(xi,mu[j],inv(precision[j])) * ntmp[j]/(alpha+n-1)
            new_j=int(max(set(c_i))+1)
            phi[i,new_j] = alpha / (alpha + n - 1) * compute_px(xi,a,B,c,m)
            phi[i] = phi[i] / np.sum(phi[i])
            cluster_i = np.where(phi[i])[0]
            diri = stats.dirichlet.rvs(phi[i][phi[i]>0])
            c_i[i] = cluster_i[np.argmax(diri)]
            if c_i[i]==new_j:
                mu[new_j], precision[new_j] = update_phi(X[c_i==new_j],a,B,c,m)
        for cluster_index in set(c_i):
            c_idx = int(cluster_index)
            mu[c_idx], precision[c_idx] = update_phi(X[c_i==c_idx],a,B,c,m)
        num_clusters.append([np.sum(c_i==int(i)) for i in set(c_i)])
    for data_list in num_clusters:
        data_list.sort(reverse=True)
    return c_i, phi, mu, precision, num_clusters

X = pd.read_csv('data.txt',header=None)
model = Gibbs_GMM(X,500)

plott = [len(list) for list in model[4]]
plt.figure(figsize=(8,6))
plt.title("cluster vs iteration")
plt.plot(range(500),plott)

max_clusters = max(plott)
plot_mat = np.zeros((len(model[4]),max_clusters))
for i,values in enumerate(model[4]):
    for j, count in enumerate(values):
        plot_mat[i,j] = count
plt.figure(figsize=(8,6))
plt.plot(plot_mat)
plt.title('observations vs iterations')