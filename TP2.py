import numpy as np
import os
from numpy.random import default_rng
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from math import sqrt,log
from scipy.stats import t,f
import pandas as pd
import subprocess
n,p = 100,2



def remplacer_recursif(chaine, ancien, nouveau):
    index = chaine.find(ancien)
    if index == -1:
        return chaine
    nouvelle_chaine = chaine[:index] + nouveau
    nouvelle_chaine += remplacer_recursif(chaine[index + len(ancien):], ancien, nouveau)
    return nouvelle_chaine



chem = os.path.abspath(__file__) 
chemin = os.path.dirname(chem)

def coursBeta(X,Y):
    Xt = X.copy()
    Xt = Xt.T
    B = Xt@X
    B= np.linalg.inv(B)
    Beta = B@Xt
    Beta = Beta@Y
    return Beta.reshape(-1,1)

R = default_rng(42).random((n,p))


X = np.eye(n,p)
K = 35 
m = 10
for k in range(n) :
    X[k,0] = R[k,0]*K*((-1)**k)
    X[k,1] = R[k,1]*((-1)**(k+1))*m


#calcul de Beta 
Beta = np.array([-2,7])
Beta =Beta.reshape(-1,1)
Y = X @ Beta
e = np.random.normal(loc=0,size=n,scale=sqrt(2)) 
e=np.reshape(e,(-1,1))
Y = Y + e

# Tracer la courbe 
x1, X1 =np.abs(X[:,0]), X[:,0]
x1,X1 = x1.reshape((-1,1)), X1.reshape((-1,1))

x2, X2 = np.abs(X[:,1]), X[:,1]
x2,X2 = x2.reshape((-1,1)), X2.reshape((-1,1))

xch = np.linspace(np.min(x1),np.max(x1),n)
#xch = np.linspace(np.min(x2),np.max(x2),n)
plt.scatter(x1,Y,label='Données réelles')
#plt.scatter(x2,Y,label='Données réelles')
ych = -2*xch
plt.plot(x1,ych, color='purple', label='Droite de régression')
#plt.plot(x2,ych, color='purple', label='Droite de régression')

plt.xlabel('Variable indépendante')
plt.ylabel('Variable dépendante')
plt.title('Nuage de points avec ligne de régression')
plt.legend()
#plt.show()


#calcul de Beta 
Beta = coursBeta(X,Y)
#print("Beta =",Beta)


#fonction lm
print(Y.shape, X.shape)
modele = sm.OLS(Y, X)
resume = modele.fit()
#print(resume.summary())

#ajouter une colonne de 1
X = np.hstack((np.ones((n,1)),X))
p= p+1 #valeur de p passe de 2 à 3
Beta2 = coursBeta(X,Y)
#print(Beta2)

Ye = X@Beta2 # Y estimé
#Residual standard error
s = np.linalg.norm(Y-Ye)
s= s**2
SE= s/(n-p)
RSE = sqrt(SE)
#print(RSE)

#Multiple R-squared
Ym = np.ones((n,1))*Y.sum()
Ym = Ym/n #valeur moyenne
a,b = (np.linalg.norm(Ye-Ym))**2, (np.linalg.norm(Y-Ym))**2
R2 = a/b
#print(R2)

#sigma Bj
B_mat = X.T@X
B_m = np.linalg.inv(B_mat)
Bj = B_m*SE 
Bjj = np.sqrt(np.diag(Bj))
Bjj = Bjj.reshape((-1,1))
#print(Bjj)

#Tj
T = Beta2/Bjj
#print(T)

#F
F = (R2)/(1-R2)
F = F*(n-p)/(p-1)
#print(F)

p_value = resume.pvalues
#print(p_value)


"""pt = t.ppf(0.95,df=n-p)
pf = f.ppf(0.95, dfn=p-1, dfd=n-p)
print(pf,pt)"""

#Données réelles
#Boston Housing
fichier = os.path.join(chemin,r"BostonHousing.csv")
fichier = remplacer_recursif(fichier,r'\\',r'/')
fichier = remplacer_recursif(fichier,r'c:',r'C:')
data = pd.read_csv(fichier)
data = data.values
x = data[:,1:]
y=data[:,0]
n=len(y)
one = np.ones((n,1))
x=np.hstack((one,x))
modele = sm.OLS(y,x)
resume = modele.fit()
Beta = coursBeta(x,y)
print(Beta[10])

#Forestfires
fichier = os.path.join(chemin,r"forestfires.csv")
fichier = remplacer_recursif(fichier,r'\\',r'/')
fichier = remplacer_recursif(fichier,r'c:',r'C:')

data = pd.read_csv(fichier)
XX = data[['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain']]
g=data['area']
y1 = np.reshape(g,(-1,1))
n,p= XX.shape
y2 = np.ones((n,1))+y1

mod = sm.OLS(y1,XX)
resum = mod.fit()
#print(resum.summary()) la pluie semble être le facteur le plus important

for i in range(n):
    y2[i][0]= log(y2[i][0])  #log(1+area)

model = sm.OLS(y2,XX)
resu = model.fit()
print(resu.summary()) #la pluie semble être le facteur le plus important également
 
