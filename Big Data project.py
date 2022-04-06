#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection Techiques (Classification Approach)

# For this project I will choose the most suitable machine learning model intended to detect credit card fraud using the following classification tools: Linear Regression, Decision Tree, Random forest , k-Nearest Neighbors(KNN), And Support Vector Classifier (SVC).

# ## Step 1: Importing the Packages 

# In this project I will primarily use Pandas for the data processing , NumPy for working with the arrays, seaborn package for data visualization,imblearn.over_sampling scikit-learn for data splitting , as well as developing and evaluating the classsification models (Linear Regression, Decision Tree, Random Forest, KNN , SVC)

# In[470]:


#Implementation of Packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier


# # Step 2: Exploratory Data Analysis

# The credit card fraud dataset has been obtained from a public repository Kaggle. It is comprised of a plethora of economic parameters of different credit card account holders.The columns "foreign_worker" , "other_parties", and"other payment plans" will be removed because they do not provide enough info to cause alert to a fraudulent or legitimate credit card account due to lack of variability. The classification aim is to predit whether an  account is fraudulent or legitimate based on the information provided.  

# In[321]:


#Importing Data
data = pd.read_csv(r"C:\Users\Misty\OneDrive\Documents\Credit Card Fraud.csv")
data.drop('foreign_worker', axis = 1, inplace = True) 
data.drop('other_parties', axis = 1, inplace = True )
data.drop('other_payment_plans', axis = 1, inplace = True)
data.head()


# In[442]:


#copy of our dataset 
df=data.copy()


# In[435]:


#Shape of the data after removing columns and rows
print("Rows    :",data.shape[0])
print("Columns :",data.shape[1])


# In[325]:


# how many columns in the dataset 
data.columns
#check for any NULL values in the dataset
data.info()


# Having null or missing values can lead to wrong predictions and bias, fortunately there are no missing values within our dataset that need to be removed.

# In[171]:


#The categorical variables in our dataset
for col in df.select_dtypes("object"):
    print(col)


# In[172]:


#The numeric variables dataset
for col in df.select_dtypes("int64"):
    print(col)


# Below I will take a look at the target variable 'class' and its distribution within our dataset.
# 'Class'-good -account is legitimate
#         bad'-account is fraudulent 

# In[173]:


df['class'].value_counts()


# In[17]:


sns.countplot(x='class', data=df)


# The target variable is very imbalanced with 700 good(legitimate) cases and 300 bad(fraud), further data exploration a few of the other categorical variables will be conducted before I balance the classes.

# In[174]:


#The means of the target variable 'class'
df.groupby('class').mean()


# Observations:
#     - Current balance as well as use of credit of accounts is a lot higher for the fraudulent accounts versus the legitimate 
#     -older accounts (cc_age) are more likely to be legitamate opposed to new accounts.
# 
# Lets calculate the means of our categorical variables for further data exploration.

# In[175]:


df.groupby('over_draft').mean()


# Observations: If credit usage is low account is most likely above two hundred dollars and not in overdraft.

# In[176]:


df.groupby('credit_history').mean()


# Observations: In comparison to our previous observation if credit usage is low it also is likely that the account holders credit history is in critical state.

# In[177]:


df.groupby('purpose').mean()


# Observations: Credit usage is low for those that would purchase retraining on their credit card and high for those that will spend it on business, used car or other 

# In[183]:


df.groupby('job').mean()


# Observations: Currrent balance and credit usage are low for unemployed/unskilled non residents and high for highly qualified/self employed/managers. 

# In[30]:


df.groupby('own_telephone').mean()


# Observations:Credit usage and current balance are higher for account holders that own telephones. 

# # Step 3: Data Vizualizations

# In[184]:


#Histogram of quantitative variables
data.hist(figsize=(20,10))
plt.show()


# Observations: The account holders in this dataset are in the age range of 30-40. 

# ###### Categorical values vs. 'class'

# In[185]:


#class vs. housing 
sns.set(rc = {'figure.figsize':(15,8)})
sns.set(font_scale = 1.2)
sns.countplot(df["housing"],hue="class",data=df,palette="Accent")
plt.title("Class vs. Housing",fontsize=15,color="Black")
plt.show()


# In[ ]:


Observations: Legitimate account holders are more likely to own housing.
    Housing status is a good predictor of the outcome variable.


# In[53]:


#class vs. purpose
sns.set(rc = {'figure.figsize':(15,8)})
sns.set(font_scale = 1.2)
sns.countplot(data["purpose"],hue="class",data=df,palette="Accent")
plt.title("Class vs. Purpose",fontsize=15,color="Black")
plt.show()


# Observations: Purpose might not be a good predictor of the outcome due to variability though there is noticably alot of legitimate account holders that purchase radio/tv's. 

# In[55]:


#Class vs. own_telephone
#plt.figure(figsize=(15,8))
sns.set(rc = {'figure.figsize':(15,8)})
sns.set(font_scale = 1.2)
sns.countplot(data["own_telephone"],hue="class",data=df,palette="Accent")
plt.show()


# own_telephone may not be a good predictor of the outcome due to very little variability. 

# In[56]:


#class vs. job
sns.set(rc = {'figure.figsize':(15,8)})
sns.set(font_scale = 1.2)
sns.countplot(data["job"],hue="class",data=df,palette="Accent")
plt.show()


# Job may be a good predictor of the outcome because legitimate account holders are usually skilled workers.

# # Step 4: Data Preparation for the classification models

# The categorical variables "employment", "overdraft", and "Average Credit Balance" all have greater or less than signs that can cause errors in our code , I decided to have them converted above into strings for better interpretation for the models to be developed.

# In[313]:


# Removing odd characters and changing certain categorical variables to strings
data['employment']=np.where(data['employment'] =='>=7', 'greater_than_seven', data['employment'])
data['employment']=np.where(data['employment'] =='1<=X<4', 'Between_one_and_four', data['employment'])
data['employment']=np.where(data['employment'] =='4<=X<7', 'Between_four_and_seven', data['employment'])
data['employment']=np.where(data['employment'] =='<1', 'Less_than_one', data['employment'])


data['over_draft']=np.where(data['over_draft'] =='<0', 'less_than_zero', data['over_draft'])
data['over_draft']=np.where(data['over_draft'] =='0<=X<200', 'Between_zero_and_two hundred', data['over_draft'])
data['over_draft']=np.where(data['over_draft'] =='>=200', 'greater_than_two_hundred', data['over_draft'])

data['Average_Credit_Balance']=np.where(data['Average_Credit_Balance'] =='>=1000', 'greater_than_one_thousand', data['Average_Credit_Balance'])
data['Average_Credit_Balance']=np.where(data['Average_Credit_Balance'] =='<100', 'less_than_one_hundred', data['Average_Credit_Balance'])
data['Average_Credit_Balance']=np.where(data['Average_Credit_Balance'] =='500<=X<1000', 'Between_five_hundred_and_one_thousand', data['Average_Credit_Balance'])
data['Average_Credit_Balance']=np.where(data['Average_Credit_Balance'] =='100<=X<500', 'Between_one_hundred_and_five_hundred', data['Average_Credit_Balance'])


# In[263]:


#importing packages 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# Creating the dummy variable for the categorical values below.

# In[264]:


cat_vars=['Average_Credit_Balance','over_draft','credit_history',
'purpose', 'employment','personal_status','property_magnitude',
'housing','job', 'own_telephone']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['Average_Credit_Balance','over_draft','credit_history',
'purpose', 'employment','personal_status','property_magnitude',
'housing','job', 'own_telephone']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[265]:


#final columns 
data_final=data[to_keep]
data_final.columns.values


# In[278]:


#re-entering dummy variables
X = df[['Average_Credit_Balance','over_draft','credit_history',
'purpose', 'employment','personal_status','property_magnitude',
'housing','job', 'own_telephone']]

X = pd.get_dummies(data=X, drop_first=True)


# In[330]:


# transform all data in input and output
print('Input', X.shape)
print(X[:5, :])
print('Output', y.shape)
print(y[:5])


# There is alot of encoding being done to account for a dataset with a mix of both numerical and categorical values that can be misinterpreted in the classification models if not properly converted. I have chosen to use ordinal encoder because it is usually used for datasets with a known relationship between variable which is valid for our dataset. I also used label encoder because it is need when a target variable needs to be encoded for classification predictive modeling. (bad/fraud=0, good/legitimate=1)

# In[331]:


#importing packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

#split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)
# ordinal encode target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)



# # Step 5: Dealing with imbalanced data

# The dataset is also on the smaller side with only 1000 entries and in order to draw sufficent conclusions free from potential bias on the majority class it would be suggested to increase our dataset with oversampling.
# Now we balance the data using the RandomOverSampler:

# In[336]:


#split the dataset into train and test and count new sets
from collections import Counter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")


# In[349]:


#balancing data by oversampling the minority class
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")


# In[403]:


#new shape of (training set)data
X_res, y_res.shape


# Now the data is balanced; I chose to over-sample only on the training data, because it can assist with making sure that none of the information in the test data is being used
# to create synthetic observations affecting the classification model interpretations.

# # Step 6: Classification Models and Performance Metrics

# In[540]:


#Import Performance metric packages
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score 


# In[532]:


#Logistic regression classification model
lr = LogisticRegression()
lr.fit(X_res, y_res)
lr_yhat = lr.predict(X_test)
# evaluate predictions
lr_pred = lr.predict(X_res)
confusion_matrix = confusion_matrix(y_res, lr_pred)
conf_matrix=pd.DataFrame(data=confusion_matrix,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_res, y_res)))


# In[535]:


#Decision Tree model
tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
tree_model.fit(X_res, y_res)
tree_yhat = tree_model.predict(X_test)
# evaluate predictions
tree_model_pred = tree_model.predict(X_res)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(tree_model.score(X_res, y_res)))
confusion_matrix = confusion_matrix(y_res, tree_model_pred)
conf_matrix=pd.DataFrame(data=confusion_matrix,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");


# In[537]:


#SVC classification model 
svm = SVC()
svm.fit(X_res, y_res)
svm_yhat = svm.predict(X_test)
# evaluate predictions
svm_pred = svm.predict(X_res)
confusion_matrix = confusion_matrix(y_res, svm_pred)
conf_matrix=pd.DataFrame(data=confusion_matrix,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");
print('Accuracy of Support vector classifier on test set: {:.2f}'.format(svm.score(X_res, y_res)))


# In[539]:


#Random Forest tree classification model
rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_res, y_res)
rf_yhat = rf.predict(X_test)
# evaluate predictions
rf_pred = rf.predict(X_res)
print('Accuracy of RandomForest classifier on test set: {:.2f}'.format(rf.score(X_res, y_res)))
confusion_matrix = confusion_matrix(y_res, rf_pred)
conf_matrix=pd.DataFrame(data=confusion_matrix,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");


# In[541]:


#Knn classification model
n = 5
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_res, y_res)
knn_yhat = knn.predict(X_test)
#evaluate predictions
knn_pred = knn.predict(X_res)
print('Accuracy of knn classifier on test set: {:.2f}'.format(knn.score(X_res, y_res)))
confusion_matrix = confusion_matrix(y_res, knn_pred)
conf_matrix=pd.DataFrame(data=confusion_matrix,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");


# Results of Confusion Matrix:
# SVC:451+392 correct predictions   35+94 incorrect 
# KNN:440 + 363 correct predictions, 46 + 123 incorrect
# DT:397+289 correct predictions, 89+ 197 incorrect 
# RF:375 + 355 correct predictions, 111+ 131 incorrect 
# LR:364+ 351 correct predictions,   135+122 incorrect

# RESULTS:
# Based on the results of our Accuracy scores the SVC model is the most accurate classification model for our data set at 87% and the decision tree is the least Accurate classification model at 71%. 
# 

# In[ ]:




