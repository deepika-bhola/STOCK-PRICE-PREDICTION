from gettext import install
from unittest import result
from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle
import opeen
import plotly.graph_objects as go
app = Flask(__name__, template_folder="templates")
app.config["IMAGE_UPLOADS"] = "static/"
@app.route("/",methods=["GET","POST"])
def home():
	return render_template("col.html")

@app.route("/predict",methods=["GET","POST"])
def gfg():
    if request.method == "POST":
       x1= request.form.get("stock")
       x=x1.upper()
       y=int(request.form.get('days'))
       def function(x):
             import pickle
             import yfinance as yf
             import pandas as pd
             import plotly.graph_objects as go
             import matplotlib.pyplot as plt
             from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
             from prophet import Prophet
             import plotly.offline as py
    
             try:
                 df=pd.read_csv("C:\\Users\\Shashank\\Desktop\\New folder\\fla_ap\\Equity.csv")
                 if x in df['Issuer Name'].unique():
                    def dataframe(x):
                         y=x+'.ns'
                         a=yf.Ticker(y)
                         stock1=a.history(period='2y',interval='1d')
                         return stock1
                 st= dataframe(x)
                 st.head()
                 st_open=pd.DataFrame(st['Open'])
                 st_open.reset_index(inplace=True)
                 st_open.rename(columns={'Date':'ds','Open':'y'},inplace=True)
                 st_m=st.reset_index()
                 fig = go.Figure(data=[go.Ohlc(x=st_m['Date'],
                 open=st['Open'], high=st['High'],
                 low=st['Low'], close=st['Close'],increasing_line_color='green',decreasing_line_color='red')])
                 py.plot(fig,filename="static/"+"filename.html",auto_open=False)
                 pickle.dump(fig, open('figg.pickle', 'wb'))
                 fig2 = plt.figure(figsize=(7.2,4.8),dpi=65)
                 plt.plot(st_open,label='Actual Price')
                 plt.legend(loc=4)
                 plt.savefig('static/ARIMA.png')
                 plt.close(fig2) 
                 model = Prophet(
                 yearly_seasonality=True,
                 weekly_seasonality=False,
                 daily_seasonality=False, 
                 interval_width=0.9
                 )
                 model.add_seasonality(
                 name='weekly', 
                 period=30.5, 
                 fourier_order=5
                 )
                 model.fit(st_open)
                 pickle.dump(model,open('model5.pkl','wb'))
                 print('done')
             except Exception as a :
                print(a)    
       function(x)
       model=pickle.load(open("model5.pkl","rb"))
       figx = pickle.load(open('figg.pickle', 'rb'))
       future = model.make_future_dataframe(periods=y,freq='d',include_history=False)
       forecast = model.predict(future)
       result=forecast.loc[:,["ds","yhat_lower","yhat","yhat_upper"]]
       final_df_1 = result.rename(columns={'yhat': 'Sales', 'ds':'Date'})
       return render_template("col.html",tables=[final_df_1.to_html(classes='forecast')],graph=figx)

   
if __name__ =="__main__":
 app.run(debug=True ,
 port=5000)
