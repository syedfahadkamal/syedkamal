#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv("SwachhBharat.csv")


years = ["2019-2020", "2020-21", "2021-22", "2022-23 (till January 31-03-2023)"]

for year in years:
    funds_released_col = f"{year} - Funds Released"
    funds_utilized_col = f"{year} - Funds Utilized"
    utilization_rate_col = f"Utilization Rate {year}"
    
    data[utilization_rate_col] = data[funds_utilized_col] / data[funds_released_col]

utilization_rate_columns = [f"Utilization Rate {year}" for year in years]
print("\nUtilization Rates:")
print(data[['State/ UT'] + utilization_rate_columns])


overall_funds_released = data[[f"{year} - Funds Released" for year in years]].sum().sum()
overall_funds_utilized = data[[f"{year} - Funds Utilized" for year in years]].sum().sum()
overall_utilization_rate = overall_funds_utilized / overall_funds_released
print("\nOverall Utilization Rate:", overall_utilization_rate)


# In[2]:



data["Difference Before 2020"] = data["2019-2020 - Funds Released"] - data["Allocation under SBM-U"]
data["Difference After 2020"] = (
    data["2020-21 - Funds Released"] + 
    data["2021-22 - Funds Released"] + 
    data["2022-23 (till January 31-03-2023) - Funds Released"]
) - data["2019-2020 - Funds Released"]


print("\nDifferences in Spending:")
print(data[['State/ UT', 'Difference Before 2020', 'Difference After 2020']])


# In[3]:


data["Increase After 2020"] = data["Difference After 2020"] > 0
data["Increase Amount After 2020"] = data["Difference After 2020"].where(data["Increase After 2020"], 0)

increased_spending_states = data[data["Increase After 2020"]][["State/ UT", "Increase Amount After 2020"]]
print("\nStates/UTs with Increased Spending after 2020 and Corresponding Increase Amount:")
print(increased_spending_states)


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="State/ UT", y="Increase Amount After 2020")
plt.xticks(rotation=90)
plt.xlabel("State/ UT")
plt.ylabel("Increase Amount After 2020")
plt.title("Increase in Spending After 2020 by State/ UT")
plt.tight_layout()
plt.show()

utilization_rate_data = data[["Utilization Rate 2019-2020", "Utilization Rate 2020-21", "Utilization Rate 2021-22", "Utilization Rate 2022-23 (till January 31-03-2023)"]]
plt.figure(figsize=(10, 6))
sns.heatmap(utilization_rate_data, annot=True, cmap="YlGnBu")
plt.xlabel("Fiscal Year")
plt.ylabel("State/ UT")
plt.title("Utilization Rates by State/ UT")
plt.tight_layout()
plt.show()


# In[5]:


data["Utilization Rate Difference Before 2020"] = data["Utilization Rate 2019-2020"] - data["Allocation under SBM-U"]
data["Utilization Rate Difference After 2020"] = (
    data["Utilization Rate 2020-21"] + 
    data["Utilization Rate 2021-22"] + 
    data["Utilization Rate 2022-23 (till January 31-03-2023)"]
) - data["Utilization Rate 2019-2020"]

print("\nUtilization Rate Differences:")
print(data[['State/ UT', 'Utilization Rate Difference Before 2020', 'Utilization Rate Difference After 2020']])


# In[6]:


import numpy as np
import statsmodels.api as sm

print("Missing or Infinite Values Check:")
print(data.isnull().sum())
print(data.isin([np.inf, -np.inf]).sum())


X_before = sm.add_constant(data["Allocation under SBM-U"])
y_before = data["Utilization Rate Difference Before 2020"]

X_before = X_before[~y_before.isna()]
y_before = y_before.dropna()

model_before = sm.OLS(y_before, X_before).fit()


X_after = sm.add_constant(data[["Utilization Rate 2020-21", "Utilization Rate 2021-22", "Utilization Rate 2022-23 (till January 31-03-2023)"]])
y_after = data["Utilization Rate Difference After 2020"]


X_after = X_after[~y_after.isna()]
y_after = y_after.dropna()

model_after = sm.OLS(y_after, X_after).fit()


print("\nRegression Analysis for Utilization Rate Differences Before 2020:")
print(model_before.summary())

print("\nRegression Analysis for Utilization Rate Differences After 2020:")
print(model_after.summary())


# In[7]:


import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.scatter(model_before.fittedvalues, y_before, color='blue', label='Before 2020')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('Predicted Utilization Rate Difference')
plt.ylabel('Actual Utilization Rate Difference')
plt.title('Regression Results Before 2020')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(model_after.fittedvalues, y_after, color='green', label='After 2020')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('Predicted Utilization Rate Difference')
plt.ylabel('Actual Utilization Rate Difference')
plt.title('Regression Results After 2020')
plt.legend()
plt.show()


# In[8]:


from sklearn.metrics import mean_squared_error

rmse_before = np.sqrt(mean_squared_error(y_before, model_before.predict(X_before)))
rmse_after = np.sqrt(mean_squared_error(y_after, model_after.predict(X_after)))


print("\nRMSE for Utilization Rate Differences Before 2020:", rmse_before)
print("RMSE for Utilization Rate Differences After 2020:", rmse_after)


# In[ ]:





# In[ ]:




