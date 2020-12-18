import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# testing huhuhuhuhuhuh



train = pd.read_csv('./application_train.csv')
print('Training data rows and columns: ', train.shape)
train.head()

test = pd.read_csv('./application_test.csv')
print('Testing data rows and columns: ', test.shape)
test.head()


# loans that were paid off(value 0) and that weren't (value 1)
train["TARGET"].value_counts() 

plt.show()
train["TARGET"].astype(int).plot.hist();





# missing vales statistics 
#missing_values = missing_values_table(train)
#missing_values.head(20)


#unique classes from each column
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

#Create a label encoder object 
labEnc = LabelEncoder()
labEnc_count = 0

#Iterate thorugh the columns
for c in train:
    if train[c].dtype =='object':
        # if 2 or fewer unqiue categories 
        if len(list(train[c].unique())) <= 2:
            # train on the training data
            labEnc.fit(train[c])
            #transform both training and testing data
            train[c] = labEnc.transform(train[c])
            test[c] = labEnc.transform(test[c])
            #keep track of how many columns were label encoded
            labEnc_count += 1
print('%d columns were label encoded.' % labEnc_count)

# one-hot encoding of categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

print('New training data rows and columns, after label encoding and one-hot encoding: ', train.shape)
print('New testing data rows and columns, after label encoding and one-hot encoding: ', test.shape)

train_labels = train['TARGET']

# Align the traingin and testing data, keep only columns of resent in both dataframes
train, test = train.align(test, join ='inner', axis =1 )

# Add the target back in 
train["TARGET"] = train_labels

print("Train rows and columns: ", train.shape)
print("Test rows and columns: ", test.shape)

# Change negative to positive 
(train["DAYS_BIRTH"]/-365).describe()
train['DAYS_EMPLOYED'].describe()
plt.show()

# Show days of employment histogram
train['DAYS_EMPLOYED'].plot.hist(title="Days Employment Histogram");
plt.xlabel("Days Employment");


# Replace the anomalous values with nan
train["DAYS_EMPLOYED"].replace({365243: np.nan},inplace =True)

plt.show()
train["DAYS_EMPLOYED"].plot.hist(title = "Days of employment");
plt.xlabel("Days of Employment");



# Find correlations with the target and sort
train_correlations = train.corr(method="pearson")['TARGET'].sort_values()
# Display correlations
print(' Positive Correlations:\n', train_correlations.tail(15))
print('\n Negative Correlations:\n', train_correlations.head(15))






plt.show()
plt.figure(figsize = (8, 6))

# KDE plot of loans that were repaid on time
sns.kdeplot(train.loc[train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

 # KDE plot of loans which were not repaid on time
sns.kdeplot(train.loc[train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.legend();





# Age information into a separate dataframe
age_data = train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)

# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()


plt.show()
plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group '); plt.ylabel('Default Rate')
plt.title('Default by Age Group');





extracted_data = train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
extracted_data_corrs = extracted_data.corr()



plt.show()
plt.figure(figsize = (8, 8))

# Heatmap of correlations
sns.heatmap(extracted_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');




plt.figure(figsize = (10, 10))
# iterate through the sources
for i, source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
    # create a new subplot for each source
    plt.subplot(3, 1, i+1)
    #plot repaid loans
    sns.kdeplot(train.loc[train["TARGET"]==0, source], label ='target == 0')
    #plot loans that were not repaid
    sns.kdeplot(train.loc[train["TARGET"]==1, source], label = 'target == 1')
    
    # label the plots
    plt.title("Distribution of %s by Target Value" %source)
    plt.xlabel('%s'% source); plt.ylabel('Density');
    plt.legend();
plt.tight_layout(h_pad=2.5)
plt.show()






from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer

# Drop the target from the training data
if 'TARGET' in train:
    train = train.drop(columns = ['TARGET'])
else:
    train = train.copy()
    

# Copy of the testing data
test = test.copy()

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

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)





from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

x = train
y = train_labels

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
# Train on the training data
log_reg.fit(x_train, y_train)

log_reg_pred = log_reg.predict_proba(x_test)[:, 1]

roc_auc_score(y_test, log_reg_pred)



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

random_forest.fit(x_train, y_train)

predictions = random_forest.predict_proba(x_test)[:, 1]

roc_auc_score(y_test, predictions)


