# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 00:25:38 2022

@author: Misty
"""
# For this project I will build a machine learning model intended to detect credit card fraud.

#Step 1: Uploading and reading Data
#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"C:\Users\Misty\OneDrive\Documents\Credit Card Fraud.csv")
data.head()


#copy
df=data.copy()

#shape of the data
print("Rows    :",data.shape[0])
print("Columns :",data.shape[1])


#Step 2: Clean Data, check for Null values and data types
# columns 
df.columns
#check for any NULL values in the dataset
df.info()
#There are no missing values 

#Check for datatypes of the attributes in the data
df.dtypes.value_counts()
#

#Step 3: Visualize Data (pie graph and histogram) and check correlations
#encoding of target qualitative variable 'class'

#Target variable is "class"; good is non-fraud, bad is fraud

#The qualitative variables
for col in df.select_dtypes("object"):
    print(col)

#visualize qualitative variables 
for col in df.select_dtypes("object"):
    plt.figure()
    df[col] .value_counts(normalize=True).plot.pie()

#The quantitative variables 
for col in df.select_dtypes("int64"):
    print(col)
    
#Histogram of quantitative variables
data.hist(figsize=(20,10))
plt.show()
     
df.corr()
    
#Correlation of the continuous variables using pearson correlation coefficient
plt.figure()
plt.title('Pearson Correlation Matrix',fontsize=23)
sns.heatmap(df.corr())

#all the categorical columns with their values -
cols=df.describe(include="O").columns
for i in cols:
    print("Distinct_values :\n 'column_name' =",i)
    print(df[i].unique())
    print("")


#Statistical Analysis of the Categorical columns 


plt.figure(figsize=(16,4))
plt.subplot(121)
sns.countplot(df["over_draft"],hue="class",data=df,palette="Blues")
plt.title("over_draft",fontsize=15,color="Black")
plt.subplot(122)
sns.scatterplot(data["credit_history"],data["credit_usage"],hue=df["class"])
plt.title("Distribution of Credit usage Vs Credit history : On basis of class",fontsize=15,color="Black")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["employment"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["foreign_worker"],hue="class",data=df,palette="Accent")
plt.show()


#plt.figure(figsize=(13,4))
sns.countplot(data["own_telephone"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["job"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["housing"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["property_magnitude"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["other_payment_plans"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["personal_status"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["Average_Credit_Balance"],hue="class",data=df,palette="Accent")
plt.show()

#plt.figure(figsize=(13,4))
sns.countplot(data["purpose"],hue="class",data=df,palette="Accent")
plt.show()

#the values in each categorical attribute
print("purpose : ",df.purpose.unique())
print("over_draft: ",df.over_draft.unique())
print("housing : ",df.housing.unique())
print("credit_history : ",df['credit_history'].unique())
print("Average_Credit_Balance : ",df['Average_Credit_Balance'].unique())
print("employment : ",df['employment'].unique())
print("personal_status : ",df['personal_status'].unique())
print("other_parties : ",df['other_parties'].unique())
print("property_magnitude : ",df['property_magnitude'].unique())
print("other_payment_plans : ",df['other_payment_plans'].unique())
print("housing : ",df['housing'].unique())
print("job: ",df['job'].unique())
print("own_telephone : ",df['own_telephone'].unique())
print("foreign_worker : ",df['foreign_worker'].unique())
print("class : ",df['class'].unique())


#Step 4: Splitting Data into Training and Testing sets (including encoding of qualitative variables)
#encoding target variable 
code={
      "good": 0,
      "bad":1}


#dummy variables for qualitative independent variables 

X = df[['foreign_worker','Average_Credit_Balance','over_draft','credit_history',
'purpose', 'employment','personal_status', 'other_parties','property_magnitude', 'other_payment_plans',
'housing','job', 'own_telephone']]

X = pd.get_dummies(data=X, drop_first=True)
X.head()

#dummy variable for qualitative dependent variable 

Y = df['class']
Y = pd.get_dummies(data=Y, drop_first=True, prefix="class_bad")
Y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#Step 5 Determining Fraud vs.non fraud distribution with SVC evaluating SVC model

# first modelling
# sklearn modules
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,Normalizer,PolynomialFeatures
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.model_selection import learning_curve,GridSearchCV

svc=make_pipeline(StandardScaler(),SVC())

# evaluation function
def evalu(model):
    model.fit(X_train,y_train)
    ypred=model.predict(X_test)
    print("confusion matrix",confusion_matrix(y_test,ypred))
    print(classification_report(y_test,ypred))
    N,train_score,val_score=learning_curve(model,X_train,y_train,scoring="f1",train_sizes=np.linspace(0.1,1,10),cv=5)
    plt.figure()
    plt.plot(N,train_score.mean(axis=1),label="train_score")
    plt.plot(N,val_score.mean(axis=1),label="validation_score")
    plt.legend()
    
evalu(svc)

