#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.mixture import BayesianGaussianMixture


# In[4]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
import plotly.express as px
from sklearn.datasets import make_blobs


# In[5]:


df = pd.read_csv('Shoes_data.csv')


# In[6]:


df['rating']=df['rating'].apply(lambda x: str(x).replace(' out of 5 stars','') if ' out of 5 stars' in str(x) else str(x))
df['price']=df['price'].apply(lambda x: str(x).replace('₹','') if '₹' in str(x) else str(x))
df['total_reviews']=df['total_reviews'].apply(lambda x: str(x).replace(' ratings','') if ' ratings' in str(x) else str(x))
df['total_reviews']=df['total_reviews'].apply(lambda x: str(x).replace(' rating','') if ' rating' in str(x) else str(x))
df['rating']=df['rating'].astype(float)
df['price']=df['price'].astype(float)
df['total_reviews']=df['total_reviews'].astype(int)


# In[7]:


sns.heatmap(df.corr(),annot=True)


# In[8]:


plt.figure(figsize=(10, 5))
sns.distplot(df[df["Shoe Type"]=="Men"].price,kde=False,rug=False)
sns.distplot(df[df["Shoe Type"]=="Women"].price,kde=False,rug=False)

plt.legend(labels=['Men', 'Women'])
plt.show()


# In[9]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x='Shoe Type', y='price', data=df, showfliers=False, ax=ax)
sns.stripplot(x='Shoe Type', y='price', data=df, jitter=True, color='black', ax=ax)

plt.show()


# In[10]:


plt.figure(figsize=(12, 12))
plt.legend(fontsize=10)
plt.tick_params(labelsize=10)
ax=sns.scatterplot(x=df['rating'],y=df['price'],hue=df['Shoe Type'],size=df['total_reviews'],data=df,sizes=(50,500))
plt.xticks(rotation=0)
ax.legend(loc='upper left',bbox_to_anchor=(1,1))


# In[11]:


import nltk
nltk.download('stopwords')
nltk.download('punkt') 


# In[12]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import string

list_stopwords = set(stopwords.words('english'))
df['reviews2'] = df['reviews'].str.lower()
df['reviews2'] = df['reviews2'].apply(word_tokenize)
df['reviews2'] = df['reviews2'].apply(lambda x: [word for word in x if word not in list_stopwords])
df['reviews2'] = df['reviews2'].apply(lambda x : [word.translate(str.maketrans('', '', string.punctuation)) for word in x])
df['reviews2'] = df['reviews2'].apply(lambda x : [word for word in x if len(word) > 1])


# In[13]:


df1=pd.DataFrame(df['reviews2'].explode())


# In[14]:


pd.set_option('display.max_rows',50)
df1.groupby('reviews2')['reviews2'].count().sort_values(ascending=False).head(50).plot.bar(figsize=(10,5))


# In[15]:


def rating_judge(ex):
    if ex <3 :
        return -1
    elif ex>3 :
        return 1
    else:
        return 0


# In[16]:


df.loc[:,'Rpolarity']=df.loc[:,'rating'].apply(rating_judge)
df


# In[17]:


df.groupby('Rpolarity')['title'].count().plot.bar()


# In[18]:


pip install vaderSentiment


# In[19]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[20]:


def rounder(num):
    return round(num)
    if num > 0: return 1
    if num < 0: return -1
    return 0


# In[21]:


analyzer = SentimentIntensityAnalyzer()


# In[22]:


Vpol = []

for text in df['reviews']:
    Vpol.append(rounder(analyzer.polarity_scores(text)['compound']))

df['VPolarity'] = Vpol 

vnotagree = df[df['Rpolarity']!=df['VPolarity']]
vagree = df[df['Rpolarity']==df['VPolarity']]

print(f"Overall length {len(df)} ")
print(f"VADER agreements/disagreements {len(vagree)}/{len(vnotagree)}")
print(f"Accuracy: {len(vagree)/len(df)*100}% ")


# In[23]:


df.head()


# In[24]:


df=df.replace({'Men':1, 'Women': 0})


# In[25]:


df_ana=df.loc[:,['price','rating','total_reviews','Shoe Type','Rpolarity','VPolarity']]


# In[26]:


pip install pycaret


# In[27]:


pip install markupsafe==2.0.1


# In[28]:


import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics


# In[29]:


import jinja2


# In[30]:


from sklearn.cluster import KMeans
import numpy as np


# In[31]:


from pycaret.clustering import *
data_clust = setup


# In[32]:


digits = load_digits()
data = scale(digits.data)
y = digits.target

k = 10
samples, features = data.shape


# In[33]:


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# In[34]:


from sklearn.preprocessing import scale
scaled_data = scale(digits.data)


# In[35]:


from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_data.shape


# In[36]:


import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import metrics


# In[37]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Shoes_data.csv')
data.head()


# In[38]:


scaler = preprocessing.StandardScaler()


# In[39]:


X = data.values[:, 1:5]
Y = data.values[:,0]


# In[40]:


from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage="average",
affinity="cosine")
model.fit(scaled_data)
print(Y2)
print(model.labels_)
print(metrics.silhouette_score(scaled_data, model.labels_))


# In[41]:


from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
aff = ["euclidean", "l1", "l2", "manhattan", "cosine"]
link = ["ward", "complete", "average"]
for a in aff:
 for l in link:
  if(l=="ward" and a!="euclidean"):
   continue
 else:
   print(a,l)
   model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
   model.fit(scaled_data)
   print(metrics.silhouette_score(scaled_data, model.labels_))


# In[42]:


from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
for k in range(2, 20):
 kmeans = cluster.KMeans(n_clusters=k)
 kmeans.fit(scaled_data)
 print(k)
 print(metrics.silhouette_score(scaled_data, kmeans.labels_))
plt.legend()
plt.show()


# In[43]:


kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(scaled_data)


# In[44]:


kmeans = cluster.KMeans(n_clusters=2, init='k-means++')
kmeans.fit(scaled_data)


# In[45]:


kmeans.inertia_


# In[46]:


SSE = []
for i in range(1,20):
    kmeans = cluster.KMeans(n_clusters = i, init='k-means++') 
    kmeans.fit(scaled_data)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker="*")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[47]:


kmeans = cluster.KMeans(n_clusters=5, init='k-means++')
kmeans.fit(scaled_data)
pred = kmeans.predict(scaled_data)
pred


# In[48]:


frame = pd.DataFrame(scaled_data)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[49]:


df = df[["reviews", "reviews_rating"]]
df.head()


# In[50]:


rew = []
rat = []
for j in df.index:
 lst = [i for i in df.iloc[j].reviews.split('||')]
 for k in lst:
  rew.append(k)

for j in df.index:
 lst = [i for i in df.iloc[j].reviews_rating.split('||')]
 for k in lst:
  rat.append(k)

df = pd.DataFrame(list(zip(rew, rat)),
 columns =['Review', 'Review_rating'])


# In[51]:


df.head()


# In[52]:


import nltk
from string import punctuation


# In[53]:


import re
def lower(text):
 return text.lower()
def remove_punctuation(text):
 return text.translate(str.maketrans('','', punctuation))
def remove_stopwords(text):
 return " ".join([word for word in str(text).split() if word not in stop])
# Removing all words with digits and standalone digits
def remove_digits(text):
 return re.sub(r'\d+', '', text)
def remove_emoji(text):
 emoji_pattern = re.compile("["
 u"\U0001F600-\U0001F64F" # emoticons
u"\U0001F300-\U0001F5FF" # symbols & pictographs
u"\U0001F680-\U0001F6FF" # transport & map symbols
u"\U0001F1E0-\U0001F1FF" # flags (iOS)
u"\U00002702-\U000027B0"
u"\U000024C2-\U0001F251"
"]+", flags=re.UNICODE)
 return emoji_pattern.sub(r'', text)
# Removing all non-printable symbols like "ड", "ட"
def remove_non_printable(text):
 text = text.encode("ascii", "ignore")
 return text.decode()
# One function to clean it all
def clean_text(text):
 text = lower(text)
 text = remove_punctuation(text)
 #text = remove_stopwords(text)
 text = remove_digits(text)
 text = remove_emoji(text)
 text = remove_non_printable(text)
 return text


# In[54]:


def get_first_digit(text):
 match = re.search(r'\d', text)
 return match[0]


# In[55]:


df['Review_rating']=df['Review_rating'].apply(get_first_digit)
df['clean_review']=df['Review'].apply(clean_text)
df.head()


# In[56]:


all_text_clean = str()
for sentence in df['clean_review'].values:
 all_text_clean += sentence
''.join(set(all_text_clean))


# In[57]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
plt.figure(figsize=(40,25))
subset1 = df[df['Review_rating']=='1']
text = subset1.clean_review.values


# In[58]:


all_text_clean = str()
for sentence in df['clean_review'].values:
 all_text_clean += sentence
''.join(set(all_text_clean))


# In[59]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
plt.figure(figsize=(40,25))
subset1 = df[df['Review_rating']=='1']
text = subset1.clean_review.values


# In[60]:


df.info()


# In[61]:


df["Review_rating"]=pd.to_numeric(df["Review_rating"])


# In[62]:


df.info()


# In[63]:


def condition(x):
 if x >= 4:
  return "Positive"
 elif x == 3:
  return "Neutral"
 else:
  return "Negative"
df['Rates'] = df['Review_rating'].apply(condition)
df


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


sns.set_style('whitegrid')
sns.countplot(x='Review_rating', data=df, palette='YlGnBu_r')


# In[66]:


sns.set_style('whitegrid')
sns.countplot(x='Rates', data=df, palette='summer')


# In[67]:


df=pd.DataFrame(df['Rates'],columns=['Rates', 'Count'])
df['Count']=1
df=df.groupby('Rates').sum()
df


# In[ ]:





# In[ ]:





# In[ ]:




