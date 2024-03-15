# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KAMALESHWAR KV
RegisterNumber:  23013347
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))


## Output:
![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/1d92ee17-2f81-4fcb-83b8-29b6d2dbc514)


![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/0ff39265-189e-4480-ad4e-817d835e47b5)


![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/62a87d5c-5933-4276-b96d-ade008d5862d)


![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/c01bd8fd-b686-4e97-979b-32af15503dbb)


![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/dae706e2-5d14-4bac-ba43-24f068e237b4)


![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/325d6aff-6b75-4e3b-922d-80db73dd25ff)


![image](https://github.com/Kamaleshwa/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144980199/33ad0e9d-7f6d-45f1-b4a2-d7c4921a3051)









## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
