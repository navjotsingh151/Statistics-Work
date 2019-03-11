#!/usr/bin/env python
# coding: utf-8

# ## Normal Distribution

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Generation
# 
# ## Male distribution:
# 
# Normal Distribution sample of 1000 records with Mean 69.5 and STD Deviation of 2.7 
# 
# ## Female distribution:
# 
# Normal Distribution sample of 1000 records with Mean 63.7 and STD Deviation of 2.3 

# In[2]:



mu_m, sigma_m = 69.5, 2.7
male = np.random.normal(mu_m, sigma_m, 1000)


mu_f, sigma_f = 63.7, 2.3
female = np.random.normal(mu_f, sigma_f, 1000)


# ### Plot Generation

# In[3]:


plt.figure(figsize=(10,10))
sns.distplot( male , color="skyblue", label="Male", kde=False)
sns.distplot( female , color="red", label="Female", kde=False)
plt.legend()
plt.show()


# ## Minimum Percentage from Top x% of the population

# In[4]:


percentage = 2.2
sample_count = 1000

required_sample = int(percentage*sample_count/100)
required_sample

# descing sorted male and Female 

male[::-1].sort()
female[::-1].sort()

print("Min Value for male height out of 2.2% highest sample :",min(male[:required_sample]))
print("Min Value for female height out of 2.2% highest sample :",min(female[:required_sample]))
# for i in range(0, required_sample):

    


# ## Quiz

# In[5]:


a =  [6.5, 3.6, 2.5, 10.1, 7.3]
mean = np.mean(a)
variance = np.var(a)
print("Variance", variance)
std = np.sqrt(variance)
print("Standard Deviation", std)


# In[ ]:




