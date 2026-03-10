import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import warnings


data = pd.read_csv('pokemon.csv')
# 3.CLEANING DATA

# DIAGNOSE DATA for CLEANING
'''
print(data.head())
print(data.tail())
print(data.columns)
print(data.shape)
print(data.info())
'''


# EXPLORATORY DATA ANALYSIS
'''
print(data['Type 1'].value_counts(dropna =False))
print(data.describe())
# VISUAL EXPLORATORY DATA ANALYSIS
data.boxplot(column='Attack',by = 'Legendary')
plt.show()
'''


# TIDY DATA
"""
data_new = data.head()
print(data_new)
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
print(melted)
# PIVOTING DATA
melted.pivot(index = 'Name', columns = 'variable',values='value')
print(melted)
"""


# CONCATENATING DATA
'''
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)
#print (conc_data_row)

data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column
print (conc_data_col)
'''


# DATA TYPES
'''
print (data.dtypes)
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
print (data.dtypes)'''


# MISSING DATA and TESTING WITH ASSERT
'''
print(data.info())
print(data["Type 2"].value_counts(dropna =False))
data1 = data
print(data1["Type 2"].dropna(inplace = True))
#assert  data['Type 2'].notnull().all()
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all()
'''

# 4. PANDAS FOUNDATION

# BUILDING DATA FRAMES FROM SCRATCH
'''
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df["capital"] = ["madrid","paris"]
df["income"] = 0 
print(df)
'''


# VISUAL EXPLORATORY DATA ANALYSIS
"""
data1 = data.loc[:,["Attack","Defense","Speed"]]
#data1.plot()
#data1.plot(subplots = True)
#data1.plot(kind = "scatter",x="Attack",y = "Defense")
#data1.plot(kind = "hist", y = "Defense", bins = 50,range = (0,250))

fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
"""


# STATISTICAL EXPLORATORY DATA ANALYSIS
'''
#print(data.describe())
# INDEXING PANDAS TIME SERIES
time_list = ["1992-03-08","1992-04-12"]
#print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
#print(type(datetime_object))

# RESAMPLING PANDAS TIME SERIES
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
#print(data2)
#print(data2.loc["1993-03-16"])
#print(data2.loc["1992-03-10":"1993-03-16"])

#print(data2.resample("A").mean(numeric_only=True))
#print(data2.resample("M").mean(numeric_only=True))
#print(data2.resample("M").first().interpolate("linear"))
#print(data2.resample("M").mean(numeric_only=True).interpolate("linear"))
'''
