#Neural network for RPL bank data set
#Importing the packages
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# creating instance for ine hot encoding
enc = OneHotEncoder()

#Loading the data set into Python
rpl = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\ann\\RPL.csv")

#Dropping the first three columns
rpl.drop(['RowNumber','CustomerId','Surname'], axis = 1)

#Converting catagorical data into numeric data
def OneHotEncoders(name,data):
    sample = data[name]
    sample=sample.to_frame()
    enc_sample = pd.DataFrame(enc.fit_transform(sample).toarray())
    enc_sample=enc_sample.iloc[:,:-1]
    data.drop([name],axis=1,inplace=True)
    data=pd.concat([data,enc_sample],axis=1)
    return data
rpl = OneHotEncoders("Geography",rpl)
rpl = OneHotEncoders("Gender",rpl)

#Creating custom function for normalization
def nor_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

#Applying normalization function on the complete data set
rpl_norm = nor_func(rpl.iloc[:,0:10])
rpl_norm.describe()

#Splitting the data into training and testing data
train,test = train_test_split(rpl_norm, test_size = 0.20)

x_train = train.drop(['Exited'], axis = 1)
y_train = train.iloc[:, 10]
x_test  = test.drop(['Exited'], axis = 1)
y_test  = test.iloc[:, 10]

num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

#Defining Neural Network model
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(6,input_dim =1,activation="relu"))
    model.add(Dense(3,activation="tanh"))
    model.add(Dense(4,activation="tanh"))
    model.add(Dense(5,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model


# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=30,epochs=20)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
# accuracy on test data set
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 


# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
# accuracy on train data set
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))