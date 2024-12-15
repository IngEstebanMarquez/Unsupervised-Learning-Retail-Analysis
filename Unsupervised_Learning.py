#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning

# ## Customer Segmentation
# <p> We will focus on understanding and working on a use case for customer segmentation. But before diving into that, here's a brief list of additional applications that can be developed using data collected from my customers:
# 
# - Descriptive Statistics
# - Customer Segmentation
# - Churn Prediction
# - Customer Lifetime Value (CLTV)
# 
# The segmentation will be based on a methodology called <b>RFM</b>
# 
# </p>

# In[3]:


# Import Pandas, Numpy, Seaborn and Matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

# Working with Dataset called "Online Retail.csv"
df = pd.read_csv("C:/Users/Josecito/Downloads/Online Retail.csv", encoding='latin1')
df.head()


# In[4]:


# Exploratory Analysis
print(df.info())
print("---------------------------------------------------")
print(df.describe())
print("---------------------------------------------------")
print("Null values:")
print(df.isnull().sum())


# ## Recency
# <p>An indicator that tells us how recent a customer's purchase is.</p>

# In[6]:


# Get unique customers
customer = df['CUSTOMER_ID'].dropna().unique()
customer


# In[7]:


# Get the last purchase date for each customer
max_purchase = df.groupby('CUSTOMER_ID')['INVOICE_DATE'].max().reset_index()
df['INVOICE_DATE'] = pd.to_datetime(df['INVOICE_DATE'], dayfirst=True)
max_purchase['INVOICE_DATE'] = pd.to_datetime(max_purchase['INVOICE_DATE'])
max_purchase


# In[8]:


# We are going to calculate our Recency metric. We will do this by subtracting the days of the last purchase date from each observation.
ref_date = df['INVOICE_DATE'].max()
max_purchase['RECENCY'] = (ref_date - max_purchase['INVOICE_DATE']).dt.days
max_purchase['RECENCY'].head()


# In[9]:


# Merge the unique customers DataFrame with the one we just created for the last purchase date
customer = pd.merge(pd.DataFrame(customer, columns=['CUSTOMER_ID']), max_purchase, on='CUSTOMER_ID')
customer


# In[10]:


# Plot a histogram of Recency
sns.histplot(customer['RECENCY'], bins=30)
plt.xlabel('Recency (days)')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# Print the Summary Statistics for Recency
print("Descriptive Statistics for Recency:")
print(customer['RECENCY'].describe())


# ## Frequency
# <p>The frequency with which a customer purchases one or more products.</p>

# In[13]:


# Getting the total number of purchases per cliente
frequency = df.groupby('CUSTOMER_ID')['INVOICE_NO'].nunique().reset_index()
frequency.columns = ['CUSTOMER_ID', 'FREQUENCY']
frequency


# In[14]:


# Merge the DataFrame we just created with the unique customers DataFrame
customer = pd.merge(customer, frequency, on='CUSTOMER_ID')
customer


# In[15]:


# Plot a histogram of Frequency
plt.figure(figsize=(10, 5))
sns.histplot(customer['FREQUENCY'], bins=55)
plt.xlabel('Number of purchases')
plt.ylabel('Frequency')
plt.show()


# In[16]:


# Print the Summary Statistics for Frequency
print("Descriptive Statistics for Frequency:")
print(customer['FREQUENCY'].describe())


# ## Monetary
# <p>The total amount a customer has spent purchasing my products.</p>

# In[18]:


# Calculate the total amount for each purchase
df['MONETARY'] = df['QUANTITY'] * df['UNIT_PRICE']

# Get the monetary value of purchases per customer
monetary = df.groupby('CUSTOMER_ID')['MONETARY'].sum().reset_index(name='MONETARY')
monetary


# In[19]:


# Merge the DataFrame we just created with the unique customers DataFrame
customer = pd.merge(customer, monetary, on='CUSTOMER_ID')
customer


# In[20]:


# Plot a histogram of Monetary
sns.histplot(customer['MONETARY'], bins=25)
plt.xlabel('Total Purchase Amount')
plt.ylabel('Frequency')
plt.show()


# In[21]:


# Print the Summary Statistics for Monetary
print("Descriptive Statistics for Monetary::")
print(customer['MONETARY'].describe())


# ## k-Means Algorithm
# <p>We have already created our main indicators for the RFM methodology. Now it's time to do <i>Machine Learning</i>. For this, we will use an unsupervised algorithm called <b>k-Means</b>.</p>

# In[23]:


# Cluster ordering function
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# ## Elbow Method
# <p>What is my optimal number of clusters? Let's build an <i>elbow chart</i> to figure it out.</p>

# In[25]:


# Import kMeans library
from sklearn.cluster import KMeans


# In[26]:


# Initial Setup - Let's use the Recency indicator as a reference
sse = {}
recency = customer[['RECENCY']]

for k in range(1, 10):
    # Instantiate the k-means algorithm iterating over k
    kmeans = KMeans(n_clusters=k, random_state=1)
    
    # Train the algorithm
    kmeans.fit(recency)
    
    # Assign the labels
    recency["clusters"] = kmeans.labels_
    
    # Append the inertia or variation to the sse array
    sse[k] = kmeans.inertia_
    
# Elbow chart
plt.figure(figsize=(12,8))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.show()


# In[27]:


# Instantiate the algorithm with 4 clusters for Recency
kmeans = KMeans(n_clusters=4, random_state=1)

# Train the algorithm
kmeans.fit(recency)

# Get the predictions
customer['RECENCY_CLUSTER'] = kmeans.labels_

# Sort the clusters
customer = order_cluster('RECENCY_CLUSTER', 'RECENCY', customer, False)

# Descriptive statistics of the created cluster
customer.groupby('RECENCY_CLUSTER')['RECENCY'].describe()


# In[28]:


# Instantiate the algorithm with 4 clusters for Frequency
kmeans = KMeans(n_clusters=4, random_state=1)

# Train the algorithm
kmeans.fit(frequency)

# Get the predictions
customer['FREQUENCY_CLUSTER'] = kmeans.labels_

# Sort the clusters
customer = order_cluster('FREQUENCY_CLUSTER', 'FREQUENCY', customer, True)

# Descriptive statistics of the clusters
print("Descriptive statistics for Frequency Cluster:")
print(customer.groupby('FREQUENCY_CLUSTER')['FREQUENCY'].describe())


# In[29]:


# Instantiate the algorithm with 4 clusters for Monetary
kmeans = KMeans(n_clusters=4, random_state=1)

# Train the algorithm
kmeans.fit(monetary)

# Get the predictions
customer['MONETARY_CLUSTER'] = kmeans.labels_

# Sort the clusters.
customer = order_cluster('MONETARY_CLUSTER', 'MONETARY', customer, False)

# Descriptive statistics of the clusters
print("Descriptive statistics for Monetary Cluster:")
print(customer.groupby('MONETARY_CLUSTER')['MONETARY'].describe())


# ## Segmentation Score
# <p>The k-means algorithm provides a generalized segmentation, but we can customize it further by creating a metric that assigns a score to the value of each cluster.</p>

# In[31]:


# Let's create our score by summing the value of each cluster
customer['SCORE'] = customer['RECENCY_CLUSTER'] + customer['FREQUENCY_CLUSTER'] + customer['MONETARY_CLUSTER']

# Get the average for each of the metrics of the created scores
customer['SCORE'].mean()


# In[32]:


# Create a function that assigns the following:
# If score <= 1 then 'Low-Value', if score >1 and <=4 then 'Average', if score >4 and <=6 then 'Potential', and finally if score >6 then 'High-Value'
def segment(score):
    if score <= 1:
        return 'Low-Value'
    elif 1 < score <= 4:
        return 'Average'
    elif 4 < score <= 6:
        return 'Potential'
    else:
        return 'High-Value'
        
# Create a column applying this function to the 'SCORE' field
customer['SEGMENT'] = customer['SCORE'].apply(segment)


# In[33]:


customer.head()


# In[34]:


# Print the proportion or total number of customers by segment
print("Proportion of customers by segment:")
print(customer['SEGMENT'].value_counts(normalize=True))


# In[35]:


# Set the 'bmh' style
plt.style.use('bmh')

# Filter the values for RECENCY < 4000
filtered_customer = customer[customer['RECENCY'] < 4000]

# Create a scatter plot of 'MONETARY' VS 'RECENCY' by Segment
sns.scatterplot(data=filtered_customer, x='RECENCY', y='MONETARY', hue='SEGMENT', s=100)
plt.title('MONETARY vs RECENCY by Segment')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend(title='Segment')
plt.show()


# In[36]:


# Create a scatter plot of 'MONETARY' vs 'FREQUENCY' by Segment
sns.scatterplot(data=filtered_customer, x='FREQUENCY', y='MONETARY', hue='SEGMENT', s=100)
plt.title('MONETARY vs FREQUENCY by Segment')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.legend(title='Segment')
plt.show()


# ### Observations:
# 1. Most customers are in the Average segment (65.8% of the total), which implies moderate purchase frequency and low spending.
# 2. There is a considerable number of customers in the Potential segment (26.6% of the total), whose behavior tends to be between that of Average customers and those classified as High-Value.
# 3. It is the smallest segment (7.6% of the total), consisting of customers who spend the most money and purchase the most frequently.
# 4. Customers in cluster 3 (with the highest purchase frequency) exhibit very active purchasing behavior, with an average of 10.28 purchases per customer, although with high dispersion (std of 16.4), which could indicate that some of these customers make extremely frequent purchases.
# 5. Customers in clusters with lower frequencies (0, 1, 2) have purchase frequencies ranging from 1 to 4, reflecting that most customers buy occasionally.
# 6. Customers in the High-Value segment not only have higher spending, but also higher purchase frequency and recency, indicating that these customers are the most valuable to the company.
# 7. The Average segment has a more moderate purchasing behavior, with average spending and lower frequency compared to High-Value customers.
# 8. The Potential segment has behavior similar to Average, but with slightly lower frequency, suggesting that these customers could improve their behavior if given an appropriate incentive.

# ### Credits and Acknowledgements
# 
# **Original code by:** Iván Alducin  
# **Modifications and Interpretations by:** Esteban Márquez  
# 
# This notebook is based on the original work by Iván Alducin, which provided the foundation for the analysis. The modifications and additional insights presented here were made by Esteban Márquez to expand upon and enhance the original approach.
# 
# Special thanks to Iván Alducin for his work, which made this analysis possible. The improvements and interpretations showcased in this notebook are my own, aimed at providing a deeper understanding of the subject matter.
# 
# Esteban Márquez  
# Date: November 20, 2024
# 

# In[ ]:




