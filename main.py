from datetime import datetime
from enum import auto
from unicodedata import decomposition
import matplotlib.pyplot as plt
import re

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from prophet import Prophet


df = pd.read_csv("covid_19_data.csv", parse_dates=["ObservationDate", "Last Update"])

df.columns = [re.sub(r"[/| ]", "", col).lower() for col in df.columns]

df_br = df.loc[(df["countryregion"] == "Brazil") & (df["confirmed"] > 0)]
fig = px.line(df_br, "observationdate", "confirmed", title="Casos Confirmados")

df_br["newcases"] = list(map(
    lambda x: 0 if (x==0) \
        else df_br["confirmed"].iloc[x] - df_br["confirmed"].iloc[x-1],
        np.arange(df_br.shape[0])
))

#px.line(df_br, x="observationdate", y="newcases", title="Growth by day")

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_br.observationdate, y=df_br.deaths, name="Dies",
               mode="lines+markers", line={"color": "green"})
)

#fig.update_layout(title="Dies by Covid19 in Brazil").show()

def total_growth(data, var, start=None, end=None):
    if start == None:
        start = data.observationdate.loc[data[var] > 0].min()
    else:
        start = pd.to_datetime(start)
        
    if end == None:
        end = data.observationdate.iloc[-1]
    else:
        end = pd.to_datetime(end)
        
    past = data.loc[data.observationdate == start, var].values[0]
    present = data.loc[data.observationdate == end, var].values[0]

    n = (end - start).days
    rate = (present/past) ** (1/n) -1
    
    return rate*100
    
t_growth = total_growth(df_br, "confirmed")

def daily_growth(data, var, start=None):
    if start == None:
        start = data.observationdate.loc[data[var] > 0].min()
    else:
        start = pd.to_datetime(start)
        
    end = df_br.observationdate.max()
    n = (end - start).days  
    
    rates = list(map(
        lambda x: (data[var].iloc[x] - data[var].iloc[x-1]) / data[var].iloc[x-1], range(1, n+1)
    ))
    
    return np.array(rates) * 100

d_growth = daily_growth(df_br, "confirmed")

#first day of the first infected person
first_day = df_br.observationdate.loc[df_br.confirmed > 0].min()

#px.line(x=pd.date_range(first_day, df_br.observationdate.max())[1:],
#        y=d_growth, title="Confirmed People Growth Rate")


df_conf = df_br.confirmed
df_conf.index = df_br.observationdate

decomposition_result = seasonal_decompose(df_conf)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

ax1.plot(decomposition_result.observed)
ax2.plot(decomposition_result.trend)
ax3.plot(decomposition_result.seasonal)
ax4.plot(df_conf.index, decomposition_result.resid)
ax4.axhline(0, linestyle="dashed", c="black")



model = auto_arima(df_conf)

graph = go.Figure(go.Scatter(x=df_conf.index, y=df_conf, name="Observed"))

graph.add_trace(go.Scatter(x=df_conf.index, y=model.predict_in_sample(), name="Predicted"))

graph.add_trace(go.Scatter(x=pd.date_range("2020-05-20", "2020-05-22"), y=model.predict(31), name="Forecast"))

graph.update_layout(title="Futuresight to cases in the next 30 days")

train = df_conf.reset_index()[:-5]

test = df_conf.reset_index()[-5:]

train.rename(columns={"observationdate": "ds", "confirmed": "y"}, inplace=True)
test.rename(columns={"observationdate": "ds", "confirmed": "y"}, inplace=True)

oracle = Prophet(growth="logistic", changepoints=["2020-03-21", "2020-03-30", "2020-04-25", 
                                                "2020-05-03", "2020-05-10"])


population = 217265543
train["cap"] = population
oracle.fit(train)

future_dates = oracle.make_future_dataframe(periods=200)
future_dates["cap"] = population
forecast = oracle.predict(future_dates)

ml = go.Figure()

ml.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name="Prediction"))
#ml.add_trace(go.Scatter(x=test.index, y=test, name="Observed - Test"))
ml.add_trace(go.Scatter(x=train.ds, y=train.y, name="Observed - Train"))
ml.update_layout(title="Prediction of Brazil's Confirmed Cases")

ml.show()