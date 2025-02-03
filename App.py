import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset.csv')

    # Fill missing values (Numerical)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

    # Fill missing values (Categorical)
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Feature Engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Total_Income_log'] = np.log(df['Total_Income'] + 1)

    # Drop unnecessary columns (KEEPING LoanAmount)
    df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Total_Income', 'Loan_ID'], inplace=True)

    # Encoding categorical features
    encoders = {}
    for col in ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Store encoder for later use

    return df, encoders

df, encoders = load_data()

# Split features and target variable
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Handle imbalanced data
oversample = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Train-test split
X_resampled_train, X_resampled_test, y_resampled_train, y_resampled_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42
)

# Train and save best model (model3)
@st.cache_resource
def train_model():
    model3 = RandomForestClassifier()
    model3.fit(X_resampled_train, y_resampled_train)
    joblib.dump(model3, "loan_model.pkl")
    return model3

model3 = train_model()

# Evaluate Model
train_acc = accuracy_score(y_resampled_train, model3.predict(X_resampled_train))
test_acc = accuracy_score(y_resampled_test, model3.predict(X_resampled_test))

# Streamlit UI
st.title("🏦 Loan Approval Prediction System")
st.subheader(f"📊 Model Accuracy: Train {train_acc:.2f}, Test {test_acc:.2f}")

st.write("Fill in the details below to check loan eligibility.")

# User Input Fields
gender = st.selectbox("Gender", encoders['Gender'].classes_)
married = st.selectbox("Married", encoders['Married'].classes_)
dependents = st.selectbox("Dependents", encoders['Dependents'].classes_)
education = st.selectbox("Education", encoders['Education'].classes_)
self_employed = st.selectbox("Self Employed", encoders['Self_Employed'].classes_)
credit_history = st.selectbox("Credit History", ["No", "Yes"])
property_area = st.selectbox("Property Area", encoders['Property_Area'].classes_)

# Selectbox for Total Income
total_income_category = st.selectbox(
    "Total Monthly Income Category",
    ["<5000", "5000-10000", "10000-20000", "20000-50000", ">50000"]
)

# Convert total income category to numerical value
income_mapping = {"<5000": 4000, "5000-10000": 7500, "10000-20000": 15000, "20000-50000": 35000, ">50000": 60000}
total_income = income_mapping[total_income_category]

# Loan Amount Slider
loan_amount = st.slider("Loan Amount (in $)", 500, 200000, 25000, step=500)

# Convert inputs to model format
# Ensure user input matches training features exactly
user_data = pd.DataFrame({
    "Gender": [encoders['Gender'].transform([gender])[0]],
    "Married": [encoders['Married'].transform([married])[0]],
    "Dependents": [encoders['Dependents'].transform([dependents])[0]],
    "Education": [encoders['Education'].transform([education])[0]],
    "Self_Employed": [encoders['Self_Employed'].transform([self_employed])[0]],
    "Credit_History": [1 if credit_history == "Yes" else 0],
    "Property_Area": [encoders['Property_Area'].transform([property_area])[0]],
    "Total_Income_log": [np.log(total_income + 1)],
    "LoanAmount": [loan_amount]
})

# 🔹 Reorder columns to match training data
user_data = user_data[X_resampled_train.columns]  

# 🔹 Convert user input to match model input type
user_data = user_data.astype(float)

# Prediction Button
if st.button("🔍 Check Loan Eligibility"):
    try:
        prediction = model3.predict(user_data)[0]
        if prediction == 1:
            st.success("✅ Congratulations! Your loan is likely to be *approved*.")
        else:
            st.error("❌ Unfortunately, your loan application may be *rejected*.")
    except Exception as e:
        st.error(f"⚠ Error: {e}")
        
st.markdown("---")
st.markdown("📌 Note: This is a machine learning model-based prediction and may not be 100% accurate.")