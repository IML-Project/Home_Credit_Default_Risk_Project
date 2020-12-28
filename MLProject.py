import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns





train = pd.read_csv('../../Data/application_train.csv')
print('Training data rows and columns: ', train.shape)
train.head()

test = pd.read_csv('../../Data/application_test.csv')
print('Testing data rows and columns: ', test.shape)
test.head()


# one-hot encoding of categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print('New training data rows and columns, after one-hot encoding: ', train.shape)
print('New testing data rows and columns, after one-hot encoding: ', test.shape)


train_labels = train['TARGET']
# Align the training and testing data, keep only columns of resent in both dataframes
train, test = train.align(test, join ='inner', axis =1 )
# Add the target back in 
train["TARGET"] = train_labels
print("Train rows and columns: ", train.shape)
print("Test rows and columns: ", test.shape)

# Replace the anomalous values with nan
train["DAYS_EMPLOYED"].replace({365243: np.nan},inplace =True)
test["DAYS_EMPLOYED"].replace({365243: np.nan},inplace =True)


#Function with basic plots
#basic_plots(train)


train.replace([np.inf, -np.inf], np.nan)
test.replace([np.inf, -np.inf], np.nan)
train, test = remove_missing_columns(train,test,20)

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import train_test_split

# Drop the target from the training data
train_labels = train['TARGET'];
train = train.drop(columns = ['TARGET'])
train_columns = train.columns 

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

train = pd.DataFrame(train,columns=train_columns)
test = pd.DataFrame(test,columns=train_columns)



x = train
y = train_labels
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)



from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(x_train, y_train)


from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
ran_for.fit(x_train, y_train)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)



from Functions import *

classifiers = [log_reg, gnb, ran_for]

plot_classifiers(classifiers, x_test, y_test)
