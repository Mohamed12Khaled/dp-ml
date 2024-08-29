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
        return np.nan  # Handle unexpected formats

    df['Size'] = df['Size'].astype(str).apply(convert_sizes)

    # Extract 'Size' unit and fill missing values
    df['Size_inNums'] = df['Size']
    df['Size_inNums'].fillna(df['Size_inNums'].mean(), inplace=True)

    # Extract date components
    df['Updated_Month'] = df['Updated'].str.split(' ').str[0]
    df['Updated_Year'] = df['Updated'].str.split(',').str[1]
    df['Updated_Day'] = df['Updated'].str.split(' ').str[1]
    df['Updated_Day'] = df['Updated_Day'].str.split(',').str[0]

    df['Updated_Month'] = pd.to_datetime(df['Updated_Month'], format='%B', errors='coerce').dt.month
    df['Updated_Year'] = pd.to_numeric(df['Updated_Year'], errors='coerce')
    df['Updated_Day'] = pd.to_numeric(df['Updated_Day'], errors='coerce')

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
    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': XGBRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if name == 'Logistic Regression':
            results[f'{name} Accuracy'] = accuracy_score(y_test, y_pred)
            results[f'{name} Confusion Matrix'] = confusion_matrix(y_test, y_pred)
            results[f'{name} Classification Report'] = classification_report(y_test, y_pred)
        else:
            results[f'{name} MAE'] = mean_absolute_error(y_test, y_pred)
            results[f'{name} MSE'] = mean_squared_error(y_test, y_pred)
            results[f'{name} R2'] = r2_score(y_test, y_pred)

    return results

# Streamlit UI
st.title("Data Processing and Model Evaluation")

st.write("**Raw data**")
df = pd.read_csv('https://raw.githubusercontent.com/Mohamed12Khaled/dp-ml/master/PlayStore_Apps.csv')
st.write(df.head())

# Preprocess the data
df_processed = preprocess_data(df)

st.write("**Processed Data**")
st.write(df_processed.head())

# Train models and evaluate
results = train_and_evaluate(df_processed)

st.write("**Model Evaluation Results**")
for key, value in results.items():
    if isinstance(value, (np.ndarray, pd.DataFrame)):
        st.write(f"{key}:")
        st.dataframe(value)
    else:
        st.write(f"{key}: {value}")
