import pandas as pd
import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
"""
Si vous n'avez pas ces modules vous pouvez les installer    avec la commande pip install (nom de module)
"""


def simpleReg(x,y) :
    n=len(x)
    A = (x*y).sum() - n*(np.mean(x))*(np.mean(y))
    A= A/((x*x).sum()-n*((np.mean(x))**2)) #formule du cours
    B= np.mean(y)-A*np.mean(x)
    ychapeau = A*x + B*(np.ones((n,1)))
    e = y-ychapeau
    RMSE = rmse = np.sqrt(np.mean((e) ** 2))
    R2 = ((e**2).sum())/(((y-np.mean(y)*np.ones((n,1)))**2).sum())
    R2 = 1- R2
    return {'A': A, 'B': B, 'ye': ychapeau, 'erreur': e, 'RMSE': RMSE, 'R2': R2 }

def predire(x,a,b):
    n=len(x)
    return a*x + b*np.ones(x.shape)

#Définition des variables
chemin = os.path.abspath(__file__) 
chemin = os.path.dirname(chemin)
fichier = os.path.join(chemin,"house-prices2.csv")
data= pd.read_csv(fichier)
Size = data['SqFt']
Size = np.reshape(Size,(-1,1))
Real_regression = data['Price']
Real_regression= np.reshape(Real_regression,(-1,1))
x, x_restant, y, y_restant = train_test_split(Size,Real_regression, test_size=0.2, random_state=42)


#Calcul du modèle Y = aX + b
#Calcul de A 
n = len(x)
print(n)  #n vaut 100

modele = LinearRegression()
modele.fit(x,y)
A = modele.coef_
A=A[0][0]
B = modele.intercept_
B=B[0]
ye = x*A + (np.ones((n,1)))*B
RMSE = sqrt(mean_squared_error(y,ye))
R2 = r2_score(y,ye)
 

"""
A,B,R2,RMSE = simpleReg(x,y)['A'],simpleReg(x,y)['B'],simpleReg(x,y)['R2'],simpleReg(x,y)['RMSE']
yp = predire(x_restant,A,B)
"""
print("A = ", A)
print("B = ", B)
print("RMSE = ", RMSE)
print("R2 = ", R2) 


yp = modele.predict(x)


#plt.scatter(x_restant, y_restant, label='Données réelles')
plt.scatter(Size, Real_regression, label='Données réelles')

# Tracer la courbe (par exemple, une ligne de régression)
#plt.plot(x_restant, yp, color='red', label='Ligne de régression')
plt.plot(Size, predire(Size,A,B), color='red', label='Ligne de régression')

plt.xlabel('Variable indépendante')
plt.ylabel('Variable dépendante')
plt.title('Régresssion lineair')
plt.legend()


superficie = np.array([10000])
aire = superficie.reshape((-1,1))
#prix=modele.predict(aire)
prix = predire(aire,A,B)
print('prix = ', prix) 
plt.show()

#Regression multiple
X = data[['Home','SqFt','Bedrooms','Bathrooms','Offers']]
print(type(x))
Y = data['Price']
Y = np.reshape(Y,(-1,1))
x,xr, y,yr=train_test_split(X,Y,test_size=28,random_state=42)

Modele = LinearRegression()
Modele = Modele.fit(x,y)
A = Modele.coef_
A=np.reshape(A,(-1,1))

ye = x@A
RMSE = mean_squared_error(y,ye)
R2 = r2_score(y,ye)
print("RMSE= ",RMSE)
print("R2 = ",R2)
