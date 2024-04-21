#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import seaborn as  sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv("C:/Users/NOKWANDA/Downloads/Mall_Customers.csv")


# In[6]:


df.head()


# # Univariate Analysis

# In[7]:


df.describe()


# In[8]:


sns.distplot(df['Annual Income (k$)'])


# In[9]:


df.columns


# In[10]:


columns = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[16]:


sns.kdeplot(x=df['Annual Income (k$)'],shade=True,hue=df['Gender']);


# In[18]:


columns = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(x=df[i],shade=True,hue=df['Gender']);


# In[19]:


columns = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i]);


# In[22]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[25]:


sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[30]:


#df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# In[32]:


df.groupby(['Gender'])[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].mean()


# In[33]:


df.corr(numeric_only=True)


# In[34]:


sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate

# In[61]:


clustering1 = KMeans(n_clusters=3)


# In[62]:


clustering1.fit(df[['Annual Income (k$)']])


# In[47]:


clustering1.labels_


# In[63]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[64]:


df['Income Cluster'].value_counts()


# In[65]:


clustering1.inertia_


# In[66]:


intertia_scores=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)
    


# In[67]:


intertia_scores


# In[60]:


plt.plot(range(1,11),intertia_scores)


# In[68]:


df.groupby('Income Cluster')[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].mean()


# # Bivariate Clustering

# In[75]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster']= clustering2.labels_
df.head()


# In[76]:


intertia_scores2=[]
for i in range (1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[82]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[100]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Spending and Income Cluster', palette='tab10')
plt.savefig('clustering_bivariate.png')


# In[85]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[86]:


df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].mean()


# # Multivariate Clustering

# In[89]:


from sklearn.preprocessing import StandardScaler


# In[90]:


scale = StandardScaler()


# In[92]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[93]:


dff.columns


# In[94]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff.head()


# In[ ]:


dff = sacle.fit_transform(dff)


# In[95]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[96]:


intertia_scores3=[]
for i in range (1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[97]:


df


# In[98]:


df.to_csv('Clustering csv')


# In[ ]:




