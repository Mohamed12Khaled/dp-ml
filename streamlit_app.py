import streamlit as st
import pandas as pd
import numpy as np

st.title('ML app')

st.info('The deployment')
df = pd.read_csv("https://raw.githubusercontent.com/Mohamed12Khaled/dp-ml/master/PlayStore_Apps.csv")
