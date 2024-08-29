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
  df
  null = null_vals[null_vals['Percent of Null Values'] > 25]
  print(len(null))
  df= df.drop(null.index, axis = 1)
  df['Size_inNums'] = df['Size'].str.split('[M,G,K,k]').str[0]
  df['Size_inNums'] = df['Size_inNums'].astype(float)
  df['Size_inNums'].mean()
  df['Size_inNums'].fillna(df['Size_inNums'].mean(), inplace=True)
  df['Size_inLetter'] = df['Size'].str.extract(r'([A-Za-z]+)')
  df['Size_inLetter'].unique()
  df['Size_inLetter'].mode()[0]
  df['Size_inLetter'].fillna(df['Size_inLetter'].mode()[0], inplace=True)
  df['Updated_Month'] = df['Updated'].str.split(' ').str[0]
  df['Updated_Year'] = df['Updated'].str.split(',').str[1]
  df['Updated_Day'] = df['Updated'].str.split(' ').str[1]
  df['Updated_Day'] = df['Updated_Day'].str.split(',').str[0]
  def convert_sizes(text):
    text = text.replace(",", "")  # Remove commas
    if text[-1] == "M":
        return float(text[:-1])    
    elif text[-1] == "k":
        return float(text[:-1]) / 1000
    elif text[-1] == "K":
        return float(text[:-1]) / 1000
    elif text[-1] == "G":
        return float(text[:-1]) * 1000
    elif text[-1] == "+":
        return 0.0
    
  df["Size"] = df["Size"].astype(str).apply(convert_sizes)
  df["Size"]
  df['Updated_Month']= pd.to_datetime( df['Updated_Month'], format='%B' ,errors='coerce')
  df['Updated_Month']=df['Updated_Month'].dt.month
  df['Updated_Year'] = df['Updated_Year'].astype(int)
  df['Updated_Day'] = df['Updated_Day'].astype(int)
  df = df.drop(columns=['Updated'], axis=1)
  df.head()
  df.info()
