## importing libraries


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
            pickle.dump(fig, open('figg .pickle', 'wb'))
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
    ## calling of data

  