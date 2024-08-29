import streamlit as st
import pandas as pd
import numpy as np

st.title('ML app')

st.info('The deployment')
with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/Mohamed12Khaled/dp-ml/master/PlayStore_Apps.csv" , na_values=['Varies with device'])
  df
  st.write('**X**')
  X_raw = df.drop('Installs', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.Installs
  y_raw
with st.expander("Model"):
  st.write("**learning and cleaning**")
  null_vals = pd.DataFrame(df.isnull().mean() * 100, columns= ['Percent of Null Values'])
  null_vals = null_vals[null_vals['Percent of Null Values'] > 0]
  null_vals
  null = null_vals[null_vals['Percent of Null Values'] > 25]
  print(len(null))
  df= df.drop(null.index, axis = 1)
