#Neural network for forest fire data set
#Importing the packages
pip install keras
pip install TensorFlow
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#Loading the data set into Python
forest = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\ann\\fireforests.csv")

#Droping first two columns
forest.drop(['month','day'], axis = 1)
forest.describe()

#Creating custom function for normalization
def nor_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

#Applying normalization function on the complete data set
forest_norm = nor_func(forest.iloc[:,0:28])
forest_norm.describe()

#Splitting the data into training and testing data
train,test = train_test_split(forest_norm, test_size = 0.20)

x_train = train.drop(['area'], axis = 1)
y_train = train.iloc[:, 6]
x_test  = test.drop(['area'], axis = 1)
y_test  = test.iloc[:, 6]

num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

#Defining Neural Network model
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(10,input_dim =27,activation="relu"))
    model.add(Dense(15,activation="tanh"))
    model.add(Dense(12,activation="tanh"))
    model.add(Dense(8,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model


# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=300,epochs=20)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
# accuracy on test data set
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 


# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
# accuracy on train data set
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))