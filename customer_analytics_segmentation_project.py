# -*- coding: utf-8 -*-

'''
Case Study

To what extent does our platform’s acquisition channel influence the learning outcomes of our students?
Are there any geographical locations where most of our students discover the platform, specifically through social media platforms like YouTube or Facebook?

You will work on real-world customer data to perform market segmentation—crucial for businesses to understand customer behavior and improve marketing efficiency. The project will involve data preprocessing,
exploratory data analysis (EDA), feature engineering, implementation of clustering algorithms, and interpretation of results. You’ll use two popular clustering techniques: k-means and hierarchical 
clustering.
'''
# Customer Analytics Segmentation Project
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

import os
import shutil

"""## Exploratory Data Analysis"""

file_path = "data/customer_segmentation_data.csv"
df = pd.read_csv(file_path)

df.describe()

df.info()

df_segmentation = df.copy()

df_segmentation.isnull().sum()

df_segmentation = df_segmentation.fillna(0)

df_segmentation.head()

df_segmentation.dtypes

"""###Correlation Matrix"""

df_segmentation.corr()

plt.figure(figsize = (12, 9))
mask = np.triu(np.ones_like(df_segmentation.corr(), dtype = bool), k = 1)
s = sns.heatmap(df_segmentation.corr(),
               annot = True,
               cmap = 'RdBu',
               vmin = -1,
               vmax = 1)

s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title('Correlation Heatmap')
plt.savefig('corr.png')
plt.show()

#No features appear too correlated with each other

"""### Data Visualization"""

plt.figure(figsize = (12, 9))

# Plotting minutes_watched and clv

plt.scatter(df_segmentation.iloc[:, 1], df_segmentation.iloc[:, 0], c = df_segmentation["region"],edgecolor = "black", cmap = "rainbow", label = "Viewers")
plt.xlabel('Minutes watched')
plt.ylabel('CLV')
plt.title('Visualization of raw data')
plt.savefig("scatter.png")
plt.legend(title = "Legend")
plt.colorbar()
plt.show()

#The large range of the minutes_watched variable shows that we probably should standardize our data to prevent skewed clustering

"""### Dummy Variables"""

df_heard_from = df_segmentation['channel']

df_countries = df_segmentation['region']

# One hot encoding

df_dummies = pd.get_dummies(df['channel'].apply(pd.Series).stack())
df_dummies = df_dummies.groupby(level=0).sum()
df = df.join(df_dummies)

segment_dummies = pd.get_dummies(df_heard_from, prefix = 'channel', prefix_sep = '_')
df_segmentation = pd.concat([df_segmentation, segment_dummies], axis = 1)

# Same approached for regions

segment_dummies_2 = pd.get_dummies(df_countries, prefix = 'country_region', prefix_sep = '_')
df_segmentation = pd.concat([df_segmentation, segment_dummies_2], axis = 1)

# dropping the channel variable as it is non-numerical and it'll not be able to perform the segmentation

df_segmentation = df_segmentation.drop(["channel"], axis = 1)

df_segmentation

# renaming the columns after adding the dummie variables
df_segmentation = df_segmentation.rename(columns = {'channel_1':'Google', 'channel_2':'Facebook', 'channel_3':'YouTube','channel_4':'LinkedIn',
                                                    'channel_5':'Twitter', 'channel_6':'Instagram', 'channel_7':'Friend', 'channel_8':'Other',
                                                    'country_region_0':'Region_0','country_region_1':'Region_1','country_region_2':'Region_2'})

df_segmentation

"""## Model Implementation

### Standardization
"""

scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)

hier_clust = linkage(segmentation_std, method = 'ward')

plt.figure(figsize = (12,9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')

dendrogram(hier_clust,
           truncate_mode = 'level',
           p = 5,
           show_leaf_counts = False,
           no_labels = True)

plt.savefig('hierarchical.png')

plt.show()

"""### K-means Clustering"""

#Keeping track of inertia_ for elbow method

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)

plt.figure(figsize = (10,8))

# Plotting the WCSS values against the number of clusters.

plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering')
plt.savefig('line_chart.png')
plt.show()

kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 42)

kmeans.fit(segmentation_std)

"""## Model Interpretation and Results"""

df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment'] = kmeans.labels_

pd.set_option('display.max_columns', 500)

df_segm_analysis = df_segm_kmeans.groupby(['Segment']).mean()
df_segm_analysis

# Grouping the dataframe by the 'Segment' column and calculating the mean for each segment.
# This will provides profile for each cluster based on the mean values of the original features.

df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment','Region_0']].groupby(['Segment']).count()
df_segm_analysis['Prop Obs'] = df_segm_analysis['N Obs'] / df_segm_analysis['N Obs'].sum()

#Here, we're trying to gain an idea of the manner in which each cluster has split the different observations, by grouping in terms of segment and taking their count and proportion
#respectively

df_segm_analysis.round(2)

df_segm_analysis.rename({0:'Instagram Explorers',
                         1:'LinkedIn Networkers',
                         2:'Friends\' Influence',
                         3:'Google-YouTube Mix',
                         4:'Anglo-Saxon Multi-Channel',
                         5:'European Multi-Channel',
                         6:'Twitter Devotees',
                         7:'Facebook Followers',
                        })

#Based on data given in the legend

df_segm_kmeans['Labels'] = df_segm_kmeans['Segment'].map({0:'Instagram Explorers',
                         1:'LinkedIn Networkers',
                         2:'Friends\' Influence',
                         3:'Google-YouTube Mix',
                         4:'White Multiple Channel',
                         5:'European Multiiple Channel',
                         6:'Twitter Devotees',
                         7:'Facebook Followers',
                        })

x_axis = df_segm_kmeans['CLV']
y_axis = df_segm_kmeans['minutes_watched']

# Setting the figure size for the scatter plot.
plt.figure(figsize = (10, 8))

# Creating a scatter plot using seaborn.
# The 'hue' parameter colors the points based on the 'Labels' column, allowing for distinction between clusters.
sns.scatterplot(x = x_axis, y = y_axis, hue = df_segm_kmeans['Labels'])

# Setting the title of the scatter plot.
plt.title('Segmentation K-means')

# Displaying the scatter plot.
plt.show()

