import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Getting the Data

df=pd.read_csv('College_Data',index_col=0)

print(df.head())

print(df.info())

print(df.describe())

#Exploratory Data Analysis

sns.scatterplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private')
plt.show()

sns.scatterplot(x='Outstate', y='F.Undergrad',data=df, hue='Private')
plt.show()

sns.set_style('whitegrid')

g=sns.FacetGrid(df, hue='Private', palette='coolwarm',aspect=2)
g.map(plt.hist, 'Outstate',bins=20,alpha=0.7)
plt.show()

plt.figure(figsize=(10,7))

g=sns.FacetGrid(df,hue='Private', aspect=2, palette='coolwarm')

g.map(plt.hist,'Grad.Rate', bins=30,alpha=0.7)
plt.show()

print(df.loc['Cazenovia College'])

#Model Creation

from sklearn.cluster import KMeans

knn=KMeans(n_clusters=2)

knn.fit(df.drop('Private',axis=1))

print(knn.cluster_centers_)
print(knn.labels_)

#Evaluation, only possible because of labels being provided

def convert(col):
    if col=='Yes':
        return 1
    if col=='No':
        return 0

df['Cluster']=df['Private'].apply(Convert)
print(df.head())

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(df['Cluster'],knn.labels_))
print(confusion_matrix(df['Cluster'],knn.labels_))
