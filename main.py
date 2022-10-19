from datetime import datetime
import re

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("covid_19_data.csv", parse_dates=["ObservationDate", "Last Update"])

print (df.tail())