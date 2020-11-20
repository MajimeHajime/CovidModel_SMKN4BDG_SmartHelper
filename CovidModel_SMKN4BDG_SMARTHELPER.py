#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import datetime


# In[2]:


# Untuk mendapatkan data terbaru, kecepatan tergantung dengan koneksi anda
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv' 
covid_data = pd.read_csv(url, index_col=0, sep=',')


# In[3]:


covid_data.head()


# In[4]:


df=covid_data.loc[covid_data['location'] == 'Indonesia']
df = df[['continent','location', 'date','new_cases','new_deaths','total_cases','total_deaths']]
df.fillna(0)
covidByDay =df.groupby(['date'])[['total_cases']].sum().sort_values('date', ascending=False)
covidByDay.head()


# In[5]:


df = df[df['total_cases'] > 10000]
list_ed = len(df)+1 #untuk mendapatkan jumlah dari hari 
list_ed


# In[6]:


# train_ed = int(list_ed * 0.8)
# train_ed //Tidak dipakai karena berganti model


# In[7]:


ar=list(range(1,list_ed))
df.insert(0,"SN",ar,True)


# In[8]:


x1 = np.array(df["SN"]).reshape(-1,1)
y = np.array(df['total_cases']).reshape(-1,1)


# In[9]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[10]:


#df.isnull().values.any() // Pengecekan NaN, redundan


# In[11]:


# Kurang mengerti hehe
print('--'*15,end ='');print('polynomial model training',end ='');print('--'*10)
for i in range(1, 70):
    polyfet = PolynomialFeatures(degree=i)
    xa = polyfet.fit_transform(x1)
    model = linear_model.LinearRegression()
    model.fit(xa,y)
    accuracy = model.score(xa,y)
    print('accuracy(R2) with degree_{} is -->  {}%'.format(i , round(accuracy*100,3)))
print('--'*45)


# In[12]:


polyfet = PolynomialFeatures(degree=7)
xa = polyfet.fit_transform(x1)
model = linear_model.LinearRegression()
model.fit(xa,y)
yp = model.predict(xa)
yact = np.array(df['total_cases'])#.reshape(-1,1)


# In[13]:


df['CASES.predicted'] = yp
df['date']=pd.to_datetime(df['date']) 
#df.set_index('date',inplace=True) // df tidak jadi dipakai untuk graphing


# In[14]:


# Pengecekan Akurasi
plt.figure(figsize=(8, 6)) 
plt.plot(yp,"--b")
plt.plot(yact,"-g")
plt.legend(['pred', 'actual'])
plt.xticks()
# plt.yticks([])
plt.title("testingaccurat", fontdict=None, loc='center')
plt.show()


# In[15]:


# 
x_fut = np.arange(30).reshape(-1,1)
xf = x_fut+x1[-1:]
y_fut = (model.predict(polyfet.transform(xf))).astype(int)
# Mengurangi dimensi dari y_fut untuk prediksi masa depan
y_flat = y_fut.flatten()
future = pd.Series(y_flat)


# In[16]:


df.set_index(df['date'], inplace=True)
last_date = df['date'].max ()


# In[17]:


# Membuat dataframe untuk dijadikan graph
futurepred = pd.DataFrame(df['date'])
futurepred.set_index('date')


# In[18]:


# Memasukan prediksi masadepan kedalam dataframe prediksi masa depan
futurepred['y_fut'] = np.nan
for fut in future:
    last_date += datetime.timedelta(days=1)
    data = {'y_fut':fut, 'date':last_date}
    futurepred = futurepred.append(pd.DataFrame(data, index=[last_date]))


# In[19]:


futurepred


# In[20]:


# Memasukan data dari df untuk di graph
futurepred['predicted'] = df['CASES.predicted']
futurepred['total'] = df['total_cases']


# In[21]:


# Plotly
fig = px.line(futurepred, x="date", y="predicted", color=px.Constant("Prediction"))
fig.add_bar(x=futurepred["date"], y=futurepred["total"], name="Total Cases")
fig.add_trace(go.Scatter(x=futurepred["date"], y=futurepred['y_fut'], name="Future Prediction", mode='lines+markers'))
fig.show()


# In[ ]:


# KKSI AI - SMKN 4 Bandung - Smart Helper
    # Pembina:
        # Ibu Tyas
    # Berdasarkan Alphabet:
        # Aldo
        # Azmi 
        # Fidaus
    # Terimakasih Pada :
        # Diwas Pandey
        # Stack Overflow
        # Our World In Data

