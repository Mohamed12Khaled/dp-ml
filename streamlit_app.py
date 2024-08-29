import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBRegressor

# Function to preprocess the data
def preprocess_data(df):
    # Remove columns with more than 25% null values
    null_vals = df.isnull().mean() * 100
    null = null_vals[null_vals > 25]
    df = df.drop(null.index, axis=1)

    # Convert 'Size' to numeric values
    df['Size_inNums'] = df['Size'].str.split('[M,G,K,k]').str[0]
    df['Size_inNums'] = df['Size_inNums'].astype(float)
    df['Size_inNums'].fillna(df['Size_inNums'].mean(), inplace=True)

    # Extract 'Size' unit and fill missing values
    df['Size_inLetter'] = df['Size'].str.extract(r'([A-Za-z]+)')
    df['Size_inLetter'].fillna(df['Size_inLetter'].mode()[0], inplace=True)

    # Convert 'Size' with units to numeric values
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

    # Extract date components
    df['Updated_Month'] = df['Updated'].str.split(' ').str[0]
    df['Updated_Year'] = df['Updated'].str.split(',').str[1]
    df['Updated_Day'] = df['Updated'].str.split(' ').str[1]
    df['Updated_Day'] = df['Updated_Day'].str.split(',').str[0]

    df['Updated_Month'] = pd.to_datetime(df['Updated_Month'], format='%B', errors='coerce')
    df['Updated_Month'] = df['Updated_Month'].dt.month
    df['Updated_Year'] = df['Updated_Year'].astype(int)
    df['Updated_Day'] = df['Updated_Day'].astype(int)

    # Drop the original 'Updated' column
    df = df.drop(columns=['Updated'], axis=1)

    # Process 'Requires Android' column
    df['Requires Android'] = df['Requires Android'].str.split('[ ,-,W,<]').str[0]
    df['Requires Android'] = df['Requires Android'].str.split('-').str[0]
    df['Requires Android'] = pd.to_numeric(df['Requires Android'], errors='coerce')
    df['Requires Android'].fillna(df['Requires Android'].mode()[0], inplace=True)

    # Replace NaN values in 'Installs'
    df['Installs'] = df['Installs'].fillna(0)
    df['Installs'] = df['Installs'].astype(str).apply(lambda x: int(x.replace(',', '').replace('+', '')))

    # Process 'Current Version' column
    df['Current Version'] = df['Current Version'].str.extract(r'(\d+\.?\d*)')
    df['Current Version'] = pd.to_numeric(df['Current Version'], errors='coerce')

    return df

# Function to train and evaluate models
def train_and_evaluate(df):
    # Drop the 'Logo' column from the Data
    df = df.drop(columns=['Logo'])

    # Define features and target
    features = df.drop(columns=['Installs'])
    target = df['Installs']

    # Convert categorical variables to numerical using LabelEncoder
    label_encoders = {}
    for column in features.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)

    gbr_model = GradientBoostingRegressor(random_state=42)
    gbr_model.fit(X_train_scaled, y_train)
    y_pred_gbr = gbr_model.predict(X_test_scaled)

    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred_logistic = logistic_model.predict(X_test_scaled)

    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)

    # Evaluation
    results = {
        'Random Forest R2': r2_score(y_test, y_pred_rf),
        'Gradient Boosting R2': r2_score(y_test, y_pred_gbr),
        'Logistic Regression Accuracy': accuracy_score(y_test, y_pred_logistic),
        'Logistic Regression Confusion Matrix': confusion_matrix(y_test, y_pred_logistic),
        'Logistic Regression Classification Report': classification_report(y_test, y_pred_logistic),
        'XGBoost MAE': mean_absolute_error(y_test, y_pred_xgb),
        'XGBoost MSE': mean_squared_error(y_test, y_pred_xgb),
        'XGBoost R2': r2_score(y_test, y_pred_xgb)
    }

    return results

# Streamlit UI
st.title("Data Processing and Model Evaluation")

with st.expander('Data'):
    st.write('**Raw data**')
    df_url = 'https://raw.githubusercontent.com/Mohamed12Khaled/dp-ml/master/PlayStore_Apps.csv'
    df = pd.read_csv(df_url)
    st.write(df.head())

# Upload CSV file if needed
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.dataframe(df.head())
    
    df_processed = preprocess_data(df)
    
    st.write("Processed Data:")
    st.dataframe(df_processed.head())
    
    results = train_and_evaluate(df_processed)
    
    st.write("Model Evaluation Results:")
    for key, value in results.items():
        if isinstance(value, (np.ndarray, pd.DataFrame)):
            st.write(f"{key}:")
            st.dataframe(value)
        else:
            st.write(f"{key}: {value}")
