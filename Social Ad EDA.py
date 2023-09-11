#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats


# In[2]:


df = pd.read_csv('/Users/dantezamora/Downloads/Social_Network_Ads.csv')

df.shape
df.head()


# Now that we have our data loaded, let's look at the distributions and data types for each column to get a better understanding our our data frame. 

# In[3]:


df.dtypes


# In[4]:


for column in df.columns:
    plt.figure(figsize = (8,5))
    sns.histplot(df[column], kde = True, bins = 10)
    plt.title(f'{column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# Since we can now see our colums distribution lets figure out the following:
#     - Avg estimated slaray
#     - Avg salary by age group
#     - Total perchases by age group
#     - Avg age

# In[5]:


AvgSalary = df['EstimatedSalary'].mean()

print(f'The avgerage estimated salary is {AvgSalary}.')


# In[6]:


AvgAge = df['Age'].mean()

print(f'The avgerage estimated salary is {AvgAge}.')


# In[7]:


#create age ranges
age_bins = [0, 18, 30, 50, float('inf')]  # Define the age group boundaries
age_labels = ['0-17', '18-30', '31-50', '51+']  # Labels for the age groups
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)


# In[8]:


AvgSalaryByAge = df.groupby('Age Group')['EstimatedSalary'].mean()

AvgSalaryByAge.plot(kind = 'bar', color = 'blue')
plt.title('Average Salary by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Estimated Salary (USD)')
plt.xticks(rotation = 45)

plt.show

AvgSalaryByAge


# In[9]:


TotalPurchasesByAge = df.groupby('Age Group')['Purchased'].sum()

TotalPurchasesByAge.plot(kind = 'bar', color = 'blue')
plt.title('Amount of Purchases by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Purchases')
plt.xticks(rotation = 45)


# From our intial data exploration we can see that the 51+ age group is the least represented but has the largest average salaries. The 31-50 age group has the largest sample size and the largest pruchase volume. 
# 
# One may conclude that the sample size of each age groups is correlated to its purchase volume - which may be true. However, we should take our analysis a few steps further finding which age groups are most probable to purchase and testing the relationships between columns. Lets start by creating a highlevel view of the relationships between colums.

# In[10]:


sns.pairplot(df.sample(100))


# In[24]:


df_matrix = df.corr()
sns.heatmap(df_matrix, annot = True, cmap='coolwarm')


# At first glance there doesn't appear to be a clear relationship between columns. Lets engineer some additional metrics to find some insights like purchase rate per age and salary groups.  

# In[11]:


PurchaseRatesByAge = df.groupby('Age Group')['Purchased'].mean()

PurchaseRatesByAge.plot(kind = 'bar', color = 'blue')
plt.title('Purchase Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Purchases')
plt.xticks(rotation = 45)

print("Purchase Rates by Age Group:")
print(PurchaseRatesByAge)


# In[12]:


chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df['Age Group'], df['Purchased']))

print("\nChi-squared test p-value:", p)


# Our Chi-squared test concludes that there is no relationship between age and purchase volume, but we do see a 90% purchase rate in the 51+ age group.

# In[20]:


#create age ranges
# Define salary groups (you can adjust the bins and labels)
salary_bins = [50000, 70000, 90000, 110000]
salary_labels = ['50-70K', '70-90K', '90-110K']
df['Salary Group'] = pd.cut(df['EstimatedSalary'], bins=salary_bins, labels=salary_labels, right=False)


# In[22]:


PurchaseRatesBySalary = df.groupby('Salary Group')['Purchased'].mean()

PurchaseRatesBySalary.plot(kind = 'bar', color = 'blue')
plt.title('Purchase Rates by Salary Group')
plt.xlabel('Salary Group')
plt.ylabel('Purchases')
plt.xticks(rotation = 45)

print("Purchase Rates by Salary Group:")
print(PurchaseRatesBySalary)


# In[23]:


chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df['Salary Group'], df['Purchased']))

print("\nChi-squared test p-value:", p)

