#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# Number of samples
num_samples = 100000

# Generate random samples from a normal distribution
activity_1 = np.random.normal(2, 2, num_samples)
activity_2 = np.random.normal(2, 2, num_samples)
activity_3 = np.random.normal(2, 2, num_samples)
activity_4 = np.random.normal(2, 2, num_samples)
final = activity_1 + activity_2 + activity_3 + activity_4

# Plot the histogram of the samples
plt.hist(activity_1, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
plt.hist(activity_2, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.hist(final, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')

# Plot the probability density function (PDF) of the normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pdf = scipy.stats.norm(2+3, (2**2 + 1.5**2)**0.5).pdf(x)
plt.plot(x, pdf, 'k--', linewidth=2)

# Add labels and a title
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')

# Show the plot
plt.show()


# In[9]:


mse = lambda a, m: ((a-m)**2).mean()
mae = lambda a, m: abs(a-m).mean()

print("activity mean", np.mean(activity_1))
print("activity mse", mse(activity_1, 2))
print("activity rmse", mse(activity_1, 2)**0.5)
print("activity mae", mae(activity_1, 2))
print("")
print("final mse", mse(final, 2*4))
print("final rmse", mse(final, 2*4)**0.5)
print("final mae", mae(final, 2*4))


# In[11]:


print(sum([2**2 for i in range(4)])**0.5)

