#!/usr/bin/env python3
'''http://pandas.pydata.org/pandas-docs/stable/10min.html '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


# -- object creation 

s = pd.Series([1,3,5,np.nan,6,8])
print(s)

dates = pd.date_range('20170207', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({ 'A' : 1.,
 'B' : pd.Timestamp('20130102'),
 'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
 'D' : np.array([3] * 4,dtype='int32'),
 'E' : pd.Categorical(["test","train","test","train"]),
 'F' : 'foo' })
print(df2)
print(df2.dtypes)

# -- view data

print(df.head())
print(df.tail(3))
print(df.index)
print(df.columns)
print(df.values)
print(df.describe()) # quick statistic summary of your data
print(df.T)
print(df.sort_index(axis=1, ascending=False)) # sort by index
print(df.sort_values(by='B')) # sort by value

# -- selection

print(df['A'])
print(df[0:3])
print(df['20170207':'20170209'])

   # selection by label
print(df.loc[dates[0]])
print(df.loc[:, ['A', 'B']])
print(df.loc['20130102':'20130104',['A','B']])
print(df.loc['20130102',['A','B']])
print(df.loc[dates[0],'A']) # scalar
print(df.at[dates[0],'A'])

   # selection by location
print(df.iloc[3])
print(df.iloc[3:5,0:2])
print(df.iloc[[1,2,4],[0,2]])
print(df.iloc[1:3,:])
print(df.iloc[:,1:3])
df.iloc[1,1] # scalar
df.iat[1,1]

   # boolean
df[df.A > 0]
df[df > 0]
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2[df2['E'].isin(['two','four'])]

   # setting values
   #...

   # missing data
   #...

   # operations
df.mean()
df.mean(1) # 1 is dimension
#df.apply(xxx)
   # ...

   # ...

   # I/O
df.to_csv('xxx.csv')
pd.read_csv('yyy.csv')
df.to_hdf('xxx.h5', 'df')
pd.read(hdf('yyy.h5', 'df'))
# to_excel

#...
