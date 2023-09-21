#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[3]:


df = pd.read_csv("CC GENERAL.csv")
df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.head(10)


# In[7]:


df = df.drop(['CUST_ID'],axis = 1)
df.head(10)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


for column in df.columns:
    plt.figure(figsize = (30,5))
    sns.histplot(df[column])
    plt.show()


# In[10]:


for column in df.columns:
    plt.figure(figsize = (30,5))
    sns.boxplot(df[column])
    plt.show()


# In[11]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

X = df['MINIMUM_PAYMENTS'].values.reshape(-1,1)
X = imputer.fit_transform(X) 

df['MINIMUM_PAYMENTS_NEW'] = X


# In[12]:


X2 = df['CREDIT_LIMIT'].values.reshape(-1,1)
X2 = imputer.fit_transform(X2) 

df['CREDIT_LIMIT_NEW'] = X2


# In[13]:


df = df.drop(['CREDIT_LIMIT','MINIMUM_PAYMENTS'],axis = 1)
df.info()


# In[14]:


df.isnull().sum().sum()


# In[15]:


sns.pairplot(df)
plt.show()


# In[16]:


plt.figure(figsize=(20,20))
corr_df = df.corr()
sns.heatmap(corr_df,annot=True)
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42)


# In[18]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
train_df = mm.fit_transform(train_df)
test_df = mm.transform(test_df)


# In[19]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
train_df = pt.fit_transform(train_df)
test_df = pt.transform(test_df)


# In[20]:


from sklearn.cluster import KMeans


# In[21]:


interclusterdistance = []

for clusters in range(1,20):
    km = KMeans(n_clusters = clusters,init ='k-means++', max_iter=300,random_state=42)
    km.fit(train_df)
    interclusterdistance.append(km.inertia_)
    
#plotting the values
plt.figure(figsize=(30,10))
plt.plot(range(1, 20), interclusterdistance, marker='o', color='r')
plt.xlabel('Number of clusters')
plt.ylabel('Inter Cluster Distance')
plt.show()


# In[22]:


km = KMeans(n_clusters = 6,init ='k-means++', max_iter=300,random_state=42)
km.fit(train_df)
y_pred = km.predict(train_df)


# In[23]:


cluster_df = pd.DataFrame(train_df,columns = df.columns)
cluster_df['clusters'] = y_pred
cluster_df.head(10)


# In[24]:


cluster_df['clusters'].value_counts()


# In[25]:


X = cluster_df[['BALANCE','PURCHASES']].to_numpy()


# In[26]:


interclusterdistance = []

for clusters in range(1,20):
    km = KMeans(n_clusters = clusters,init ='k-means++', max_iter=300,random_state=42)
    km.fit(X)
    interclusterdistance.append(km.inertia_)
    
#plotting the values
plt.figure(figsize=(30,10))
plt.plot(range(1, 20), interclusterdistance, marker='o', color='g')
plt.xlabel('Number of clusters')
plt.ylabel('Inter Cluster Distance')
plt.show()


# In[27]:


km = KMeans(n_clusters = 4,init ='k-means++', max_iter=300,random_state=42)
km.fit(X)
y_balance_pred = km.predict(X)


# In[28]:


plt.scatter(X[y_balance_pred==0, 0], X[y_balance_pred==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_balance_pred==1, 0], X[y_balance_pred==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_balance_pred==2, 0], X[y_balance_pred==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_balance_pred==3, 0], X[y_balance_pred==3, 1], s=100, c='yellow', label ='Cluster 4')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=300, c='cyan', label = 'Centroids')
plt.show()


# In[29]:


from sklearn.cluster import DBSCAN


# In[30]:


dbscan = DBSCAN(eps=2,min_samples=6)
dbscan.fit(train_df)
y_dbscan_pred = dbscan.labels_
y_dbscan_pred


# In[31]:


dbscan_df = pd.DataFrame(train_df,columns = df.columns)
dbscan_df['clusters'] = y_dbscan_pred
dbscan_df.head(10)


# In[32]:


dbscan_df['clusters'].value_counts()


# In[33]:


X = dbscan_df[['BALANCE','PURCHASES']].to_numpy()


# In[34]:


dbscan = DBSCAN(eps=0.075,min_samples=2)
dbscan.fit(X)
y_dbscan_pred = dbscan.labels_
y_dbscan_pred


# In[35]:


dbscan_df['clusters'] = y_dbscan_pred
dbscan_df['clusters'].value_counts()


# In[36]:


plt.figure(figsize=(10,10))
plt.scatter(dbscan_df['BALANCE'],dbscan_df['PURCHASES'],c=dbscan_df['clusters'])
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()


# In[37]:


get_ipython().system('pip install minisom')


# In[38]:


from minisom import MiniSom


# In[39]:


som = MiniSom(x = 20,y = 20, input_len = 17, sigma=0.25) 


# In[40]:


som.train_random(train_df, 10000) 


# In[41]:


from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(train_df):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# In[42]:


df = pd.read_csv("CC GENERAL.csv")
df.info()


# In[43]:


df.head()


# In[44]:


df.info()


# In[45]:


df.describe()


# In[46]:


df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(), inplace = True)
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(), inplace = True)


# In[47]:


df['CUST_ID'].value_counts().unique()


# In[48]:


df.drop(columns=['CUST_ID'],inplace = True)


# In[49]:


df['TENURE'].unique()


# In[50]:


X = df.iloc[:,:].values


# In[51]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[52]:


from sklearn.cluster import KMeans
kmeans = KMeans(random_state=42)


# In[53]:


from yellowbrick.cluster import KElbowVisualizer
methods = ['distortion', 'silhouette', 'calinski_harabasz']
for i in methods:
    print(i)
    visualizer = KElbowVisualizer(kmeans, k=(2,20), metric = i, timings = False)
    visualizer.fit(X)
    visualizer.show()


# In[54]:


kmeans = KMeans(n_clusters = 9, init = 'k-means++', random_state = 42)
y_means = kmeans.fit_predict(X)
y_means


# In[ ]:


plt.figure(figsize=(15,15))
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel(' Distances')
plt.show()


# In[ ]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[77]:


x


# In[62]:


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


# In[67]:


model = LinearRegression()


# In[68]:


model.fit(x, y)
LinearRegression()


# In[69]:


model = LinearRegression().fit(x, y)


# In[71]:


r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")


# In[72]:


print(f"intercept: {model.intercept_}")


# In[73]:


print(f"slope: {model.coef_}")


# In[74]:


y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")


# In[75]:


x


# In[ ]:




