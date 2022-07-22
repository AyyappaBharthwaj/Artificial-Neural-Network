#Neural network for concrete data set
#Importing the packages

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#Loading the data set into Python
concrete = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\ann\\concrete.csv")

#Creating custom function for normalization
def nor_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

#Applying normalization function on the complete data set
concrete_norm = nor_func(concrete.iloc[:,0:8])
concrete_norm.describe()

#Splitting the data into training and testing data
train,test = train_test_split(concrete_norm, test_size = 0.20)

x_train = train.iloc[:, -1]
y_train = train.iloc[:, 8]
x_test  = test.iloc[:, -1]
y_test  = test.iloc[:, 8]

num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

#Defining Neural Network model
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(3,input_dim =7,activation="relu"))
    model.add(Dense(4,activation="tanh"))
    model.add(Dense(2,activation="tanh"))
    model.add(Dense(3,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model


# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=400,epochs=20)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
# accuracy on test data set
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 


# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
# accuracy on train data set
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))