import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklear.preprocessing import RobustScalar
from sklearn.neighbours  import KNeighborsRegressor 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from matplotlib import pyplot as plt


df = pd.read_csv('LCZ14.csv')
df.drop(['Time'],1, inplace= True)



features = np.array(df.drop(['Temp'],1))
target = np.array (df['Temp'])


#Feature Transformation aand Transformation
rb = RobustScalar()

X = rb.fit_tansform(features)
y = rb.fit_transform(target)

## KNN regression 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

knn = KNeighborsRegressor (n_neighbors= 4)
knn.fit(X_train, Y_train)

Y_pred_train = knn.predict(X_train) #predictions on training data
Y_pred = knn.predict(X_test) #predictions on testing data

plt.scatter(Y_test,Y_pred)
plt.xlabel("Actual Temperature: $Y_i$")
plt.ylabel("Predicted temperature: $\hat{Y}_i$")
plt.title("Predicted Temperature vs Actual Temperature : $Y_i$ vs $\hat{Y}_i$")
plt.show()


#print('Intercept term: ',knn.intercept_) # This gives us the intercept term
#print('Coefficients: \n',knn.coef_) # This gives us the coefficients (in the case of this model, just one coefficient)
accuracy = knn.score(X_test, Y_test)
print(accuracy)

## Random forest Regression 

rf =  RandomForestRegressor()

Y_pred_train_r = rf.predict(X_train) #predictions on training data
Y_pred_r = rf.predict(X_test) #predictions on testing data

plt.scatter(Y_test,Y_pred_r)
plt.xlabel("Actual Temperature: $Y_i$")
plt.ylabel("Predicted temperature: $\hat{Y}_i$")
plt.title("Predicted Temperature vs Actual Temperature : $Y_i$ vs $\hat{Y}_i$")
plt.show()




#Mean of both the values

k1 = list(map(int,Y_pred ))


r1 = list(map(int, Y_pred_train_r ))



final = (np.add(k1,r1))/2

plt.scatter(Y_test,final)
plt.xlabel("Actual Temperature: $Y_i$")
plt.ylabel("Predicted temperature: $\hat{Y}_i$")
plt.title("Predicted Temperature vs Actual Temperature : $Y_i$ vs $\hat{Y}_i$")
plt.show()





