from datetime import datetime
import matplotlib as plt
import re

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

#open the sheet and set some columns to date
df = pd.read_csv("covid_19_data.csv", parse_dates=["ObservationDate", "Last Update"])

#rename all columns to get a default template
df.columns = [re.sub(r"[/| ]", "", col).lower() for col in df.columns]

#filtering just brazilian cases and plotting them
df_br = df.loc[(df["countryregion"] == "Brazil") & (df["confirmed"] > 0)]
fig = px.line(df_br, "observationdate", "confirmed", title="Casos Confirmados")

#creating the column to show the growth rate everyday and putting in a graphic
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

#growth rate function
def g_rate(data, var, start=None, end=None):
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
    
t_growth = g_rate(df_br, "confirmed")

#daily growth rate function
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

