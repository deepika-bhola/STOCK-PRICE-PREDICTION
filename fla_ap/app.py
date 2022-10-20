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
print('hey')
@app.route("/",methods=["GET","POST"])
def home():
	return render_template("col.html")

@app.route("/predict",methods=["GET","POST"])
def gfg():
    if request.method == "POST":
       x1= request.form.get("stock")
       x=x1.upper()
       y=int(request.form.get('days'))
       opeen.function(x)
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
