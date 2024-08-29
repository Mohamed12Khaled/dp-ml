import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
st.title('ML app')

st.info('The deployment')
df = pd.read_csv("https://raw.githubusercontent.com/Mohamed12Khaled/dp-ml/master/PlayStore_Apps.csv")
