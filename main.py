from datetime import datetime
import re

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


#open the sheet and set some columns to date
df = pd.read_csv("covid_19_data.csv", parse_dates=["ObservationDate", "Last Update"])

#rename all columns to get a default template
df.columns = [re.sub(r"[/| ]", "", col).lower() for col in df.columns]

#filtering just brazilian cases and plotting them
df_br = df.loc[(df["countryregion"] == "Brazil") & (df["confirmed"] > 0)]
fig = px.line(df_br, "observationdate", "confirmed", title="Casos Confirmados")


df_br["newcases"] = list(map(
    lambda x: 0 if (x==0) \
        else df_br["confirmed"].iloc[x] - df_br["confirmed"].iloc[x]-1,
        np.arange(df_br.shape[0])
))