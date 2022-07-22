
############################################problem 1##################################
# Import necessary libraries for MLP and reshaping the data structres
import pandas as pd
import numpy as np

#loading the dataset
Startups = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Artificial Neural Network/50_Startups.csv")
        
#details of rpl 
Startups.info()
Startups.describe()

#data types        
Startups.dtypes

#checking for na and null value
Startups.isna().sum()
Startups.isnull().sum()

#checking unique value for each columns
Startups.nunique()

#Performing EDA
EDA ={"column ": Startups.columns,
      "mean": Startups.mean(),
      "median":Startups.median(),
      "mode":Startups.mode(),
      "standard deviation": Startups.std(),
      "variance":Startups.var(),
      "skewness":Startups.skew(),
      "kurtosis":Startups.kurt()}

EDA

#variance for each column
Startups.var() 

#graphical repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(Startups,hue='Profit')

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(Startups.iloc[:,:3])

#categorical
enc_df = pd.get_dummies(Startups.iloc[:,[3]])

#final dataframe
model_df = pd.concat([Startups.iloc[:,[4]],df,enc_df], axis =1)
model_df.describe()

np.random.seed(10)

from sklearn.model_selection import train_test_split
model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data

x_train = model_df_train.iloc[:,1:].values.astype("float32")
y_train = model_df_train.iloc[:,0].values.astype("float32")
x_test = model_df_test.iloc[:,1:].values.astype("float32")
y_test = model_df_test.iloc[:,0].values.astype("float32")

#building model
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,100,100),activation='relu', max_iter=20 , solver = 'lbfgs')
model.fit(x_train,y_train)

# Evaluating the model on test data using mean absolute square error
from sklearn import metrics
mae1 = metrics.mean_absolute_error(y_test, model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

########################################Problem 2###############################
# Import necessary libraries for MLP and reshaping the data structres
import pandas as pd
import numpy as np

#loading the dataset
fireforests = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Artificial Neural Network/fireforests.csv")

#dropping unwanted column
fireforests.drop(["day","month"], axis = 1, inplace = True)
    
#details of rpl 
fireforests.info()
fireforests.describe()

#data types        
fireforests.dtypes

#checking for na and null value
fireforests.isna().sum()
fireforests.isnull().sum()

#checking unique value for each columns
fireforests.nunique()

#Performing EDA
EDA ={"column ": fireforests.columns,
      "mean": fireforests.mean(),
      "median":fireforests.median(),
      "mode":fireforests.mode(),
      "standard deviation": fireforests.std(),
      "variance":fireforests.var(),
      "skewness":fireforests.skew(),
      "kurtosis":fireforests.kurt()}

EDA

#variance for each column
fireforests.var() 

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
out = fireforests['area']
inp = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
       'dayfri', 'daymon', 'daysat', 'daysun', 'daythu', 'daytue', 'daywed',
       'monthapr', 'monthaug', 'monthdec', 'monthfeb', 'monthjan', 'monthjul',
       'monthjun', 'monthmar', 'monthmay', 'monthnov', 'monthoct', 'monthsep']

df = norm_func(fireforests[inp])

#final dataframe
model_df = pd.concat([out,df], axis =1)
model_df.describe()

np.random.seed(10)

from sklearn.model_selection import train_test_split
model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data

x_train = model_df_train.iloc[:,1:].values.astype("float32")
y_train = model_df_train.iloc[:,0].values.astype("float32")
x_test = model_df_test.iloc[:,1:].values.astype("float32")
y_test = model_df_test.iloc[:,0].values.astype("float32")

#building model
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(10,10,),activation='tanh', max_iter=10)
model.fit(x_train,y_train)

# Evaluating the model on test data using mean absolute square error
from sklearn import metrics
mae1 = metrics.mean_absolute_error(y_test, model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

############################################Problem 3####################3
# Import necessary libraries for MLP and reshaping the data structres
import pandas as pd
import numpy as np

#loading the dataset
concrete = pd.read_csv("E:/Data Science 18012022/Artificial Neural Network/concrete.csv")
        
#details of rpl 
concrete.info()
concrete.describe()

#data types        
concrete.dtypes

#checking for na and null value
concrete.isna().sum()
concrete.isnull().sum()

#checking unique value for each columns
concrete.nunique()

#Performing EDA
EDA ={"column ": concrete.columns,
      "mean": concrete.mean(),
      "median":concrete.median(),
      "mode":concrete.mode(),
      "standard deviation": concrete.std(),
      "variance":concrete.var(),
      "skewness":concrete.skew(),
      "kurtosis":concrete.kurt()}

EDA

#variance for each column
concrete.var() 

#graphical repersentation 
##historgam and scatter plot
import seaborn as sns

#boxplot for every columns
sns.boxplot("cement", data =concrete)
sns.boxplot("slag", data = concrete)
sns.boxplot("ash", data = concrete)
sns.boxplot("water", data = concrete)
sns.boxplot("superplastic", data =concrete)
sns.boxplot("coarseagg", data =concrete)
sns.boxplot("fineagg", data =concrete)
sns.boxplot("age", data =concrete)

concrete.boxplot(column=['cement', 'slag', 'ash', 'water','superplastic' , 'coarseagg' , 'fineagg' , 'age'])  

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(concrete.iloc[:,:8])

#final dataframe
model_df = pd.concat([concrete.iloc[:,[8]],df], axis =1)
model_df.describe()

np.random.seed(10)

from sklearn.model_selection import train_test_split
model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data

x_train = model_df_train.iloc[:,1:].values.astype("float32")
y_train = model_df_train.iloc[:,0].values.astype("float32")
x_test = model_df_test.iloc[:,1:].values.astype("float32")
y_test = model_df_test.iloc[:,0].values.astype("float32")

#building model
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(10,10,),activation='tanh', max_iter=10)
model.fit(x_train,y_train)

# Evaluating the model on test data using mean absolute square error
from sklearn import metrics
mae1 = metrics.mean_absolute_error(y_test, model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

#################################problem 4#########################
# Import necessary libraries for MLP and reshaping the data structres
import pandas as pd
import numpy as np

#loading the dataset
rpl = pd.read_csv("E:/Data Science 18012022/Artificial Neural Network/RPL.csv")
        
#details of rpl 
rpl.info()
rpl.describe()

#data types        
rpl.dtypes

#checking for na and null value
rpl.isna().sum()
rpl.isnull().sum()


#checking unique value for each columns
rpl.nunique()

#Performing EDA
EDA ={"column ": rpl.columns,
      "mean": rpl.mean(),
      "median":rpl.median(),
      "mode":rpl.mode(),
      "standard deviation": rpl.std(),
      "variance":rpl.var(),
      "skewness":rpl.skew(),
      "kurtosis":rpl.kurt()}

EDA

#variance for each column
rpl.var() 

rpl.drop(["RowNumber","CustomerId","Surname"], axis = 1, inplace = True)

#graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(rpl.iloc[:,[0,3,4,5,9,10]],hue='Exited')

#boxplot for every columns
sns.boxplot(x = "Exited", y = "CreditScore", data =rpl)
sns.boxplot(x = "Exited", y = "Age", data = rpl)
sns.boxplot(x = "Exited", y = "Balance", data = rpl)
sns.boxplot(x = "Exited", y = "EstimatedSalary", data = rpl)
sns.boxplot(x = "Exited", y = "Tenure", data =rpl)

rpl.boxplot(column=['CreditScore', 'Age', 'Balance', 'EstimatedSalary','Tenure'])  

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(rpl.iloc[:,[0,3,4,5,6,7,8,9]])
df.describe()
#categorical
enc_df = pd.get_dummies(rpl.iloc[:,[1,2]])
enc_df.columns

#final dataframe
model_df = pd.concat([rpl.iloc[:,[10]],df,enc_df], axis =1)

#building model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense
from keras.utils import np_utils

np.random.seed(10)

from sklearn.model_selection import train_test_split

model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data

x_train = model_df_train.iloc[:,1:].values.astype("float32")
y_train = model_df_train.iloc[:,0].values.astype("float32")
x_test = model_df_test.iloc[:,1:].values.astype("float32")
y_test = model_df_test.iloc[:,0].values.astype("float32")

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =13,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(120,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=500,epochs=20)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)

# accuracy on test data set
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

# Evaluating the model on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)

# accuracy on train data set 
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
#######################################END################################### 