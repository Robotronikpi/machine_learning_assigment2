import numpy as np
import matplotlib.pyplot as plt
import random
from math import *
import pylab as py
##general parameters
a_0 = 1
b_0 = 2
lambda_0  = 2
mu_0 = 2
N= 10
## parameters  gaussian plot
#tau = np.random.gamma(a_0,b_0,1)
#mu  = np.random.normal(mu_0,1.0/(lambda_0*tau[0]),1)
mu = np.array([1])
tau = np.array([1])
Y=np.random.normal(mu,(1.0/tau),N)
epsilon  = 0.001
taille  =400
list_tau = np.linspace(0, 2*tau[0], taille)
list_mu = np.linspace(0, 3*mu[0], taille)-0*mu[0]

def normalGamma_PDF(x,tau,mu,lambdaa,a,b):
    return (((b**a)*sqrt(lambdaa))/(gamma(a)*sqrt(np.pi*2)))*(tau**(a-0.5))*np.exp(-0.5*lambdaa*tau*(x-mu)**2)*np.exp(-b*tau)


def normalGamma(lambda_0,mu_0,a_0,b_0,Y,N):
    mean  = (lambda_0*mu_0 + N*np.mean(Y))/(lambda_0+N)
    lambdaa = lambda_0 + N
    alpha =  a_0 +N/2
    beta  = b_0 + 0.5*(N*np.var(Y)+(lambda_0*N*((np.mean(Y)-mu_0)**2))/(lambda_0+N))
    return mean,lambdaa,alpha,beta

def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def pdf_gamma(x,a,b):
    result = ((x**(a-1))*(b**a)*np.exp(-b*x))/gamma(a)
    return ((x**(a-1))*(b**a)*np.exp(-b*x))/gamma(a)

def muN(lambda_0,mu_0,N,Y):
    return (lambda_0*mu_0 + N*np.mean(Y))/(lambda_0+N)

def LambdaN(lambda_0,N,a_N,b_N):
    return (lambda_0+N)*(a_N/b_N)

def aN(a_0,N):
    return a_0 +(N+1)/2
def bN(lambda_0,N,b_0,mu_N,lambda_N,Y):
    return b_0 + 0.5*((lambda_0+N)*((1./lambda_N)+mu_N**2)-2*(lambda_0*mu_0+np.sum(Y))*mu_N+np.sum(Y**2)+lambda_0*mu_0**2)





mu_N = muN(lambda_0,mu_0,N,Y)
a_N = aN(a_0,N)
lambda_N = random.random()
lambda_N_past = random.random()
b_N = random.random()
b_N_past = random.random()
## EM estimator
while(abs(b_N_past-b_N)>epsilon or abs(lambda_N-lambda_N_past)>epsilon):
    b_N_past = b_N
    lambda_N_past = lambda_N
    b_N = bN(lambda_0,N,b_0,mu_N,lambda_N,Y)

    print("b",b_N)
    lambda_N = LambdaN(lambda_0,N,a_N,b_N)
    print("l",lambda_N)
    print(abs(b_N_past-b_N))

esti_post  = normalGamma(lambda_0,mu_0,a_0,b_0,Y,N)
X = np.linspace(-20, 20, 10000)
plt.plot(X, gaussian(X,mu[0],1./tau[0]))
plt.plot(X, gaussian(X,mu_N,b_N/a_N))
plt.plot(X, gaussian(X,esti_post[0],esti_post[3]/esti_post[2]))
plt.show()
diff_mu = abs(esti_post[0]-mu_N)
diff_v = abs(esti_post[1]-b_N/a_N)
relativ_exact =abs(esti_post[0]-mu[0])/mu[0]
relativ =  abs(b_N/a_N-mu[0])/(1./tau[0])
relativ_mu = abs(mu_N-mu[0])/mu[0]
relativ_exact_v = abs(esti_post[1]-mu[0])/(1./tau[0])
print("diff_mu = ",diff_mu)
print("diff_v = ",diff_v)
print("relative_exate  = ",relativ_exact)
print("relativ_mu = ",relativ_mu)
print("relativ = ",relativ)
print("relative_exacte_v  = ",relativ_exact_v)
print(relativ-relativ_exact_v)

## plot graph
plt.figure()
result_exact = np.zeros((taille,taille))
result =  np.zeros((taille,taille))
param = normalGamma(lambda_0,mu_0,a_0,b_0,Y,N)
for i in range(taille):
    for j in range(taille):
        result_exact[i,j]= log(abs(normalGamma_PDF(list_mu[i],list_tau[j],param[0],param[1],param[2],param[3])+0.000001))
        result[i,j] = log(abs(pdf_gamma(list_tau[j],a_N,b_N)*gaussian(list_mu[i],mu_N,1./lambda_N)+0.000001))
py.pcolor(list_tau, list_mu, result_exact)
plt.plot(tau,mu,'o',color = 'r')
plt.ylabel('mean (mu_x)')
plt.xlabel("tau_x")
titre = "Exact posterior Normal-gamma distribution, parameters : nb points = "+str(N) + " ,mu = "+ str(mu[0])+" ,tau = " + str(tau[0])+ " ,mu_0 = " + str(mu_0) + " ,a_0 = "+str(a_0) +" ,b_0 = "+str(b_0)+ " ,lambda_0 = "+ str(lambda_0)
plt.title(titre)
plt.plot(tau,mu,'o',color = 'r',label = "parameters of the point-generating Gaussian (mu, tau)")
CS = plt.contour(list_tau, list_mu, result_exact,colors=('k',))
plt.legend(loc="upper right")
#plt.clabel(CS, inline=1, fontsize=10)
plt.show()

plt.figure()
titre = "posterior Normal-gamma distribution, variational inference, parameters : nb points = "+str(N) + " ,mu = "+ str(mu[0])+" ,tau = " + str(tau[0])+ " ,mu_0 = " + str(mu_0) + " ,a_0 = "+str(a_0) +" ,b_0 = "+str(b_0)+ " ,lambda_0 = "+ str(lambda_0)
plt.ylabel('mean (mu_x)')
plt.title(titre)
plt.xlabel("tau_x")
plt.plot(tau,mu,'o',color = 'r',label = "parameters of the point-generating Gaussian (mu, tau)")
py.pcolor(list_tau, list_mu, result)
CS = plt.contour(list_tau, list_mu, result,colors=('k',))
#plt.clabel(CS, inline=1, fontsize=10)
plt.legend(loc="upper right")
plt.show()


plt.figure()
choice_tau = 20 ## choice cut 1
titre = "exact posterior Normal-gamma distribution and variational inference, parameters : nb points = " +str(N) + " ,mu = "+ str(mu[0])+" ,tau = " + str(tau[0])+ " ,mu_0 = " + str(mu_0) + " ,a_0 = "+str(a_0) +" ,b_0 = "+str(b_0)+ " ,lambda_0 = "+ str(lambda_0)+ " ,tau_x = " + str(list_tau[choice_tau])
plt.ylabel(' log of posterior probability + constant')
plt.title(titre)
plt.xlabel("mu_x")
plt.plot(list_mu, result_exact.T[choice_tau]-np.mean(result_exact.T[choice_tau]), label = "exact posterior")
plt.plot(list_mu, result.T[choice_tau]-np.mean(result.T[choice_tau]),label = "VI posterior")
plt.legend(loc="upper right")
plt.show()


plt.figure()
choice_tau = 120 ## choice cut 2
titre = "posterior Normal-gamma distribution, variational inference, parameters : nb points = "+str(N) + " ,mu = "+ str(mu[0])+" tau = " + str(tau[0])+ " ,mu_0 = " + str(mu_0) + " ,a_0 = "+str(a_0) +" ,b_0 = "+str(b_0)+ " ,lambda_0 = "+ str(lambda_0)+ " ,tau_x = " + str(list_tau[choice_tau])
plt.ylabel(' log of posterior probability + constant')
plt.title(titre)
plt.xlabel("mu_x")
plt.plot(list_mu, result_exact.T[choice_tau]-np.mean(result_exact.T[choice_tau]), label = "exact posterior")
plt.plot(list_mu, result.T[choice_tau]-np.mean(result.T[choice_tau]),label = "VI posterior")
plt.legend(loc="upper right")
plt.show()