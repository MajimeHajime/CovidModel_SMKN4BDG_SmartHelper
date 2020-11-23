from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd 
import datetime


# Untuk mendapatkan data terbaru, kecepatan tergantung dengan koneksi anda
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv' 
covid_data = pd.read_csv(url, index_col=0, sep=',')
df = covid_data.loc[covid_data['location'] == 'Indonesia']
df = df[['continent','location', 'date','new_cases','new_deaths','total_cases','total_deaths']]
df.fillna(0)

df = df[df['total_cases'] > 10000]
list_ed = len(df)+1 #untuk mendapatkan jumlah dari hari 
ar=list(range(1,list_ed))
df.insert(0,"SN",ar,True)
x1 = np.array(df["SN"]).reshape(-1,1)
y = np.array(df['total_cases']).reshape(-1,1)

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
polyfet = PolynomialFeatures(degree=7)
xa = polyfet.fit_transform(x1)
model = linear_model.LinearRegression()
model.fit(xa,y)

yp = model.predict(xa)
yact = np.array(df['total_cases'])#.reshape(-1,1)
df['CASES.predicted'] = yp
df['date']=pd.to_datetime(df['date']) 
x_fut = np.arange(30).reshape(-1,1)
xf = x_fut+x1[-1:]
y_fut = (model.predict(polyfet.transform(xf))).astype(int)
# Mengurangi dimensi dari y_fut untuk prediksi masa depan
y_flat = y_fut.flatten()
future = pd.Series(y_flat)
df.set_index(df['date'], inplace=True)
last_date = df['date'].max ()

# Membuat dataframe untuk dijadikan graph
futurepred = pd.DataFrame(df['date'])
futurepred.set_index('date')
# Memasukan prediksi masadepan kedalam dataframe prediksi masa depan
futurepred['y_fut'] = np.nan
for fut in future:
    last_date += datetime.timedelta(days=1)
    data = {'y_fut':fut, 'date':last_date}
    futurepred = futurepred.append(pd.DataFrame(data, index=[last_date]))
# Memasukan data dari df untuk di graph
futurepred['predicted'] = df['CASES.predicted']
futurepred['total'] = df['total_cases']


#export graph
futurepred.to_csv('testfile.csv')

#  ++=========================================++
#  || KKSI AI - SMKN 4 Bandung - Smart Helper ||
#  ||-----------------------------------------||
#  ||   Pembina:                              ||
#  ||       Ibu Tyas                          ||
#  ||   Anggota(Berdasarkan Alphabet):        ||
#  ||       Aldo Fakhry                       ||
#  ||       Firdaus Haqiqi                    ||
#  ||       Muhammad Azmi                     ||
#  ||-----------------------------------------||
#  ||   Terimakasih Pada:                     ||
#  ||       # Sentdex                         ||
#  ||       # Pandas Documentation            ||
#  ||       # Diwas Padley                    ||
#  ||       # Stack Overflow                  ||
#  ||       # Our World In Data               ||
#  ++=========================================++