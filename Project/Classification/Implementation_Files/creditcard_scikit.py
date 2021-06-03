# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 23:58:45 2020

@author: gouth
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
pd.set_option('display.max_columns', None)

credit_df = pd.read_csv('creditcard.csv')
credit_df.info()

#Defining the dataframe Schema, data types
print("Defining the dataframe Schema,data types")
print(credit_df.info())

#Summary statistics on every column
print("Summary Statistics on every column of dataframe")
print(credit_df.describe())

#
print("Null Values Count of every column:\n",credit_df.isnull().sum())

#calculating the correlation values
cor_df = credit_df.corr()
cor_df['Class']
sns.heatmap(cor_df)

cols =  ['V1', 'V2', 'V3', 'V4', 'V5','V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
print("boxplots of numeric columns before applying transformations")
with PdfPages('boxplots_before_scikit.pdf') as pdf:
    for i in cols:
        plt.figure()
        b = sns.boxplot(credit_df[i])
        plt.title(i)
        pdf.savefig(b.get_figure())
        plt.clf()

#creating the barplot target value counts of each category:
print("creating the barplot target value counts of each category:")
print(credit_df.Class.value_counts())
sns.countplot(credit_df.Class)
plt.savefig('barplot_scikit.pdf')



#standardscaler transformations on the numerica columns with std,mean
scaler = StandardScaler()
scaler.fit(credit_df[cols])
transformed_df = scaler.transform(credit_df[cols])
transformed_df

X_train, X_test, y_train, y_test = train_test_split(transformed_df, credit_df.Class, test_size=0.3, random_state=42)

#Model building
from sklearn.linear_model import LogisticRegressionCV
cv = LogisticRegressionCV(cv = 5,random_state=42).fit(X_train,y_train)
test_pred = cv.predict(X_test)



average_precision = average_precision_score(y_test, test_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
