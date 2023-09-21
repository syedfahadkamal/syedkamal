#!/usr/bin/env python
# coding: utf-8

# In[82]:


#First, we import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

#Then, we load datasets
df_vaccinations = pd.read_csv("country_vaccinations.csv")
df_manufacturer = pd.read_csv("country_vaccinations_by_manufacturer.csv")


# In[83]:


# Introducing the dataset for clarity and inaccuracies
print("COVID-19 World Vaccination Progress Dataset")
print("=")
print("The dataset provides information on the global vaccination progress against COVID-19.")
print("It consists of two main files: 'country_vaccinations.csv' and 'country_vaccinations_by_manufacturer.csv'.")
print("'country_vaccinations.csv' includes daily vaccination data and statistics by different vaccines used in various countries.")
print("'country_vaccinations_by_manufacturer.csv' provides details on the number of vaccine doses delivered by different manufacturers.")
print("\n")

# Showcasing 'country_vaccinations.csv' dataset
print("Head of 'country_vaccinations.csv' dataset:")
print("============")
print(df_vaccinations.head())
print("\n")


# In[84]:


# Dataset Graph
print("Dataset Summary:")
print("")
df_summary = df_vaccinations[['country', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].dropna()
df_summary = df_summary.groupby('country').sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(df_summary['country'], df_summary['total_vaccinations'])
plt.xlabel('Country')
plt.ylabel('Total Vaccinations')
plt.title('Total Vaccinations by Country')
plt.xticks(rotation=90)
plt.show()


# In[85]:


# Importing plotly to visualise using Treemap, since the above visualisations don't provide clarity
import plotly.express as px

fig = px.treemap(df_summary, path=['country'], values='total_vaccinations',
                  title='Total Vaccinations by Country (Treemap)',
                  color='total_vaccinations',
                  color_continuous_scale='Blues')
fig.show()


# In[86]:


# Bivariate analysis - Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_summary['people_vaccinated'], df_summary['people_fully_vaccinated'])
plt.xlabel('People Vaccinated')
plt.ylabel('People Fully Vaccinated')
plt.title('Bivariate Analysis: People Vaccinated vs. People Fully Vaccinated')
plt.show()


# In[87]:


# Identifiying key challenges and its description
print("Challenges to be Addressed:")
print("=====")
print("1. Noisy Data: The dataset may contain errors, inconsistencies, and missing values.")
print("2. Data Completeness: Different countries have varying levels of reporting accuracy, leading to incomplete data entries.")
print("\n")

# Summarising through statistics
print("Summary Statistics:")
print("=")
total_vaccinations = df_vaccinations['total_vaccinations'].sum()
people_vaccinated = df_vaccinations['people_vaccinated'].sum()
people_fully_vaccinated = df_vaccinations['people_fully_vaccinated'].sum()
print("Total number of vaccinations worldwide:", total_vaccinations)
print("Total number of people vaccinated worldwide:", people_vaccinated)
print("Total number of people fully vaccinated worldwide:", people_fully_vaccinated)
print("\n")


# In[88]:


# Visualising Summary Statistics through Bar graph - 
statistics = ['Total Vaccinations', 'People Vaccinated', 'People Fully Vaccinated']
counts = [total_vaccinations, people_vaccinated, people_fully_vaccinated]
plt.figure(figsize=(10, 6))
plt.bar(statistics, counts)
plt.xlabel('Statistics')
plt.ylabel('Counts')
plt.title('Summary Statistics of Global Vaccination Progress')
plt.show()


# In[89]:


# Pie chart - Summary Statistics
plt.figure(figsize=(10, 6))
plt.pie(counts, labels=statistics, autopct='%1.1f%%')
plt.title('Summary Statistics of Global Vaccination Progress (Pie Chart)')
plt.show()


# In[90]:


# Performing unsupervised analysis through K-means clustering
print("Unsupervised Analysis - K-means Clustering:")
print("=")
df_cluster = df_vaccinations[['people_vaccinated', 'people_fully_vaccinated']].dropna()
X = df_cluster.values
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
df_cluster['cluster'] = kmeans.labels_

# Visualizations, first through scatter plot with clusters
plt.scatter(df_cluster['people_vaccinated'], df_cluster['people_fully_vaccinated'], c=df_cluster['cluster'], cmap='viridis')
plt.xlabel('People Vaccinated')
plt.ylabel('People Fully Vaccinated')
plt.title('K-means Clustering of Vaccination Progress')
plt.show()

print("Head of 'country_vaccinations.csv' dataset with clusters:")
print("===========================================")
print(df_cluster.head())
print("\n")

# Visualization with a correlation heatmap
print("Correlation Heatmap:")
print("=")
plt.figure(figsize=(10, 8))
sns.heatmap(df_vaccinations.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[91]:


# Moving on to supervised analysis by conducting Linear regression
print("Supervised Analysis - Linear Regression:")
print("=")
df_regression = df_vaccinations.merge(df_manufacturer, left_on=['country', 'date'], right_on=['location', 'date'], how='inner').dropna(subset=['daily_vaccinations_raw', 'daily_vaccinations'])
X_reg = df_regression[['daily_vaccinations_raw', 'daily_vaccinations']]
y_reg = df_regression['total_vaccinations_x']
regression_model = LinearRegression()
regression_model.fit(X_reg, y_reg)

# Visualization with scatter plot with regression line
plt.scatter(X_reg['daily_vaccinations_raw'], y_reg)
plt.plot(X_reg['daily_vaccinations_raw'], regression_model.predict(X_reg), color='red')
plt.xlabel('Daily Vaccinations (Raw)')
plt.ylabel('Total Vaccinations')
plt.title('Linear Regression of Daily Vaccinations vs. Total Vaccinations')
plt.show()

print("Coefficients:", regression_model.coef_)
print("Intercept:", regression_model.intercept_)
print("\n")


# In[92]:


# Final reflections on methodology
print("Reflection on Methods Used:")
print("-")
print("The summary statistics provided a snapshot of the global vaccination progress.")
print("K-means clustering helped identify clusters of countries with similar vaccination patterns, offering insights into successful vaccination campaigns and potential challenges.")
print("Linear regression allowed us to analyze the relationship between daily vaccination numbers and overall vaccination progress, providing insights into the effectiveness of vaccination efforts.")
print("\n")


# In[93]:


#Summarised Results
numeric_results = {
    "Total Vaccinations Worldwide": total_vaccinations,
    "Total People Vaccinated Worldwide": people_vaccinated,
    "Total People Fully Vaccinated Worldwide": people_fully_vaccinated,
    "Linear Regression Coefficients": regression_model.coef_,
    "Linear Regression Intercept": regression_model.intercept_
}

print("\nNumeric Results:")
print("========")
for key, value in numeric_results.items():
    print(key + ":", value)


# In[94]:


# References 
print("References:")
print("-------")
print("COVID-19 World Vaccination Progress Dataset. Retrieved from Kaggle: https://www.kaggle.com/gpreda/covid-world-vaccination-progress")


# In[ ]:





# In[ ]:





# In[ ]:




