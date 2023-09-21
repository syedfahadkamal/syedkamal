#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv("Increment.csv")

total_new_registrations = data["Number of New Registrations"].sum()

data["Percentage of New Registrations"] = (data["Number of New Registrations"] / total_new_registrations) * 100

print("Total Number of New Registrations:", total_new_registrations)
print("\nData with Percentage of New Registrations:")
print(data)

X = sm.add_constant(data.index)  
y = data["Number of New Registrations"]


model = sm.OLS(y, X).fit()

print("\nRegression Analysis Results:")
print(model.summary())


plt.figure(figsize=(8, 6))
plt.bar(data["Year"], data["Percentage of New Registrations"], color='skyblue')
plt.xlabel("Year")
plt.ylabel("Percentage of New Registrations")
plt.title("Percentage of New Registrations Over Years")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data_2018 = pd.read_csv("BOCW2018.csv")

data_2022 = pd.read_csv("BOCW2022.csv")

data_2018["Cess Difference (Rs in Crore)"] = data_2018["Amount of cess collected (Rs. in Crore)"] - data_2018["Amount of cess spent (Rs. in Crore)"]
data_2022["Cess Difference (Rs in Crore)"] = data_2022["Cess Collected Cumulative (Rs in Crore)"] - data_2022["Expenditure Cumulative (Rs in Crore)"]


print("BOCW2018 Data with Differences:")
print(data_2018)

print("\nBOCW2022 Data with Differences:")
print(data_2022)


X_2018 = sm.add_constant(data_2018["No. of workers registered with the Board"])
y_2018 = data_2018["Cess Difference (Rs in Crore)"]

model_2018 = sm.OLS(y_2018, X_2018).fit()


X_2022 = sm.add_constant(data_2022["No. of Registered BOCW Workers"])
y_2022 = data_2022["Cess Difference (Rs in Crore)"]

model_2022 = sm.OLS(y_2022, X_2022).fit()

print("\nRegression Analysis Results for BOCW2018:")
print(model_2018.summary())

print("\nRegression Analysis Results for BOCW2022:")
print(model_2022.summary())


# In[6]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


data_2018 = pd.read_csv("BOCW2018.csv")


data_2022 = pd.read_csv("BOCW2022.csv")


data_2018["Cess Difference (Rs in Crore)"] = data_2018["Amount of cess collected (Rs. in Crore)"] - data_2018["Amount of cess spent (Rs. in Crore)"]
data_2022["Cess Difference (Rs in Crore)"] = data_2022["Cess Collected Cumulative (Rs in Crore)"] - data_2022["Expenditure Cumulative (Rs in Crore)"]


plt.figure(figsize=(10, 6))
plt.bar(data_2018["Name of the States/UTs"], data_2018["Cess Difference (Rs in Crore)"], color='blue')
plt.xlabel("States/UTs")
plt.ylabel("Cess Difference (Rs in Crore)")
plt.title("Difference in Spending (BOCW2018)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(data_2022["States/UTs"], data_2022["Cess Difference (Rs in Crore)"], color='green')
plt.xlabel("States/UTs")
plt.ylabel("Cess Difference (Rs in Crore)")
plt.title("Difference in Spending (BOCW2022)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




