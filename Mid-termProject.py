#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st

# This is the dataset, 2 of them, I would like to know, what factor affects the delivery time the most? 
# 
# 1 of all, IDA, scan through the document and see if there is any missing values. drop them for now.

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


nyc = pd.read_csv("NYC_food_order.csv",na_values=['Not given'])
fdt = pd.read_csv("Food_Delivery_Times.csv")


# In[4]:


row6 = nyc.iloc[:, 6]
print(row6)
allnumberrow6 = row6.dropna()


# In[35]:


nyc = nyc.drop(nyc.columns[:2], axis=1)
numeric_df = nyc.select_dtypes(include='number')
nyc_correlation = numeric_df.corr()
sns.heatmap(
    nyc_correlation,
    annot=True,          # 在热图上显示相关系数值
    cmap='coolwarm',     # 颜色主题
    center=0,            # 颜色中心值
    fmt='.1f',           # 数值格式（保留两位小数）
    linewidths=0.5       # 格子线宽度
)
plt.title('Correlation Heatmap', fontsize=15)


# In[36]:


print(nyc.info())
print(numeric_df.head())


# wow, this is shocking. not a single bit of correlation. Now, we need to do a bit further analysis. I did not take into consideration the difference between the weekday and weekends, I need to separate that first. 

# In[6]:


weekday_nyc = nyc[nyc['day_of_the_week'] == 'Weekday']
weekend_nyc = nyc[nyc['day_of_the_week'] == 'Weekend']
numeric_columns = nyc.select_dtypes(include=['number']).columns.tolist()
weekday_numeric = weekday_nyc[numeric_columns]
weekend_numeric = weekend_nyc[numeric_columns]


# In[7]:


weekday_corr = weekday_numeric.corr()
sns.heatmap(weekday_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Weekday Correlation Heatmap')


# In[8]:


weekend_corr = weekend_numeric.corr()
sns.heatmap(weekend_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Weekday Correlation Heatmap')


# so now, it finally shows some correlation. However, it is still too weak, there is only a little correlation between the rating of the restaurant vs the delivery time during weekdays, but even that is still weak.
# 
# Now, we are going to separate the dish categories and do another correlation map. 

# In[9]:


eastern_cuisines = ['Chinese', 'Korean', 'Japanese', 'Indian', 'Thai']
eastern = nyc[nyc['cuisine_type'].isin(eastern_cuisines)]

western_cuisines = ['Italian', 'American', 'Mediterranean', 'Middle Eastern', 'Mexican', 'Southern', 'French', 'Spanish']
western = nyc[nyc['cuisine_type'].isin(western_cuisines)]

numeric_columns = nyc.select_dtypes(include=['number']).columns.tolist()

eastern_numeric = eastern[numeric_columns]
western_numeric = western[numeric_columns]


# In[10]:


eastern_corr = eastern_numeric.corr()
sns.heatmap(eastern_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Eastern Cuisine Correlation Heatmap')


# In[11]:


western_corr = western_numeric.corr()
sns.heatmap(western_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Western Cuisine Correlation Heatmap')


# still, very low correlation. it does seem that the restaurants delivery time has nothing to do with type of restaurant.
# 
# now, we try no. 2: Impututation. What if we impute some data, and then will there be correlation? 
# 
# Fist, let's decide if the missing value is MAR, MCAR or MNAR.

# In[12]:


sns.heatmap(nyc.isna(), cmap="magma")
plt.title("Heatmap")


# this is a Univariate missing, I need to now determine if it is MNAR,MAR or MCAR.

# In[13]:


nyc["was_NaN"] = False
nyc.loc[nyc["rating"].isnull() == True, "was_NaN"] = True
nyc.head(8)


# In[14]:


sns.pairplot(nyc, hue="was_NaN")


# this is definitely MCAR.

# In[15]:


from sklearn.impute import KNNImputer
numeric_cols = nyc.select_dtypes(include=['number']).columns.tolist()
print("\n数值列列表:", numeric_cols)

imputer = KNNImputer(n_neighbors=5)

nyc[numeric_cols] = imputer.fit_transform(nyc[numeric_cols])

print(f"\nKNN填充后评分列缺失值数量: {nyc['rating'].isna().sum()}")

correlation_matrix = nyc[numeric_cols].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    square=True,
    linewidths=0.5
)

plt.title('Correlation Heatmap (After KNN Imputation)', fontsize=12)


# After imputation, it seems still has no correlation at all with ratings! 

# Next, we are going to do the same with another dataset: Food_delivery_time, see if we have the same result.

# In[16]:


fdt = pd.read_csv("Food_Delivery_Times.csv")


# In[17]:


fdt.describe()


# In[18]:


numeric_fdt = fdt.select_dtypes(include='number')
fdt_correlation = numeric_fdt.corr()
sns.heatmap(
    fdt_correlation,
    annot=True,          # 在热图上显示相关系数值
    cmap='coolwarm',     # 颜色主题
    center=0,            # 颜色中心值
    fmt='.2f',           # 数值格式（保留两位小数）
    linewidths=0.5       # 格子线宽度
)
plt.title('Correlation Heatmap', fontsize=15)


# In[19]:


sns.heatmap(fdt.isna(), cmap="magma")
plt.title("Heatmap")


# In[20]:


fdt["was_NaN"] = False
fdt.loc[nyc["rating"].isnull() == True, "was_NaN"] = True
fdt.head(8)


# In[21]:


sns.pairplot(fdt, hue="was_NaN")


# In[22]:


from sklearn.impute import KNNImputer
numeric_colsfdt = fdt.select_dtypes(include=['number']).columns.tolist()
print("\n数值列列表:", numeric_cols)

imputer = KNNImputer(n_neighbors=5)

fdt[numeric_colsfdt] = imputer.fit_transform(fdt[numeric_colsfdt])

print(f"\nKNN填充后评分列缺失值数量: {fdt['Courier_Experience_yrs'].isna().sum()}")

correlation_matrixfdt = fdt[numeric_colsfdt].corr()

sns.heatmap(
    correlation_matrixfdt,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    square=True,
    linewidths=0.5
)

plt.title('Correlation Heatmap (After KNN Imputation)', fontsize=12)


# imputation does not matter here, the only thing that matters the most is the distance in km, and has something to do with the preparation time too. 
# 
# next thing, gonna separate the categorical stats, see their interaction with the delivery time now.

# In[23]:


categorical_cols = ['Time_of_Day', 'Vehicle_Type',"Traffic_Level","Weather"]
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y='Delivery_Time_min', data=fdt)
    plt.title(f'Delivery Time vs {col}')
    plt.xticks(rotation=45)  # 防止标签重叠
    plt.show()


# now, we will do some EDA. It does seem that the only thing that is strongly correlated to delivery time is distance, therefore, we will do some analysis on distance and time.

from scipy.stats import pearsonr

plt.figure(figsize=(10, 6))
sns.scatterplot(data=fdt, x='Distance_km', y='Delivery_Time_min')
plt.title('Delivery Time and Distance')
plt.xlabel('Distance(km)')
plt.xticks(rotation=45)
plt.ylabel('Delivery Time(min)')
plt.show()

correlation, p_value = pearsonr(fdt['Distance_km'], fdt['Delivery_Time_min'])

from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear_func, fdt['Distance_km'], fdt['Delivery_Time_min'])
a, b = popt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=fdt, x='Distance_km', y='Delivery_Time_min', alpha=0.6, label='data point')

x_range = np.linspace(fdt['Distance_km'].min(), fdt['Distance_km'].max(), 100)
y_fit = linear_func(x_range, a, b)

plt.plot(x_range, y_fit, 'r-', linewidth=2, label='line: y = 3x + 26.3')
plt.legend()
plt.show()

print(f'Pearson correlation between Distance and Delivery Time：{correlation:.2f}，p Value：{p_value:.2f}')

fdt['distance_bin'] = pd.cut(fdt['Distance_km'], bins=range(0, int(fdt['Distance_km'].max()) + 5, 5))

bin_stats = fdt.groupby('distance_bin')['Delivery_Time_min'].agg(['mean', 'median','std']).reset_index()

print(bin_stats)




