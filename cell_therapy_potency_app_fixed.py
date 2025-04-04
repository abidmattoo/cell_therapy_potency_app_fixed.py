
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🔬 Cell Therapy Potency Predictor")
st.markdown("Predict **Potency (%)** based on assay and process parameters.")

# Sidebar inputs
st.sidebar.header("🧪 Input Parameters")
passage_number = st.sidebar.selectbox("Passage Number", [1, 2, 3, 4])
MOI = st.sidebar.slider("MOI", 2.0, 10.0, 5.0)
culture_days = st.sidebar.slider("Culture Days", 7, 14, 10)
transduction_eff = st.sidebar.slider("Transduction Efficiency (%)", 30.0, 90.0, 60.0)
viability = st.sidebar.slider("Viability (%)", 80.0, 98.0, 90.0)
activation_marker = st.sidebar.slider("Activation Marker (%)", 40.0, 95.0, 70.0)
IL2 = st.sidebar.slider("IL-2 Expression (pg/mL)", 100.0, 1000.0, 500.0)
IFNg = st.sidebar.slider("IFN-g Expression (pg/mL)", 100.0, 1200.0, 600.0)

# Input dataframe
input_data = pd.DataFrame({
    "passage_number": [passage_number],
    "MOI": [MOI],
    "culture_days": [culture_days],
    "transduction_efficiency": [transduction_eff],
    "viability_percent": [viability],
    "activation_marker_percent": [activation_marker],
    "IL2_expression": [IL2],
    "IFNg_expression": [IFNg],
})

# Simulate training dataset
np.random.seed(456)
n_samples = 500
train_df = pd.DataFrame({
    "passage_number": np.random.randint(1, 5, n_samples),
    "MOI": np.random.uniform(2, 10, n_samples),
    "culture_days": np.random.randint(7, 15, n_samples),
    "transduction_efficiency": np.random.uniform(30, 90, n_samples),
    "viability_percent": np.random.uniform(80, 98, n_samples),
    "activation_marker_percent": np.random.uniform(40, 95, n_samples),
    "IL2_expression": np.random.uniform(100, 1000, n_samples),
    "IFNg_expression": np.random.uniform(100, 1200, n_samples)
})
noise = np.random.normal(0, 2, n_samples)
train_df["potency_percent"] = (
    0.01 * train_df["IL2_expression"] +
    0.008 * train_df["IFNg_expression"] +
    0.1 * train_df["activation_marker_percent"] +
    0.05 * train_df["transduction_efficiency"] +
    0.05 * train_df["viability_percent"] +
    (-1.0 * train_df["passage_number"]) +  # Add passage number impact
    noise
).clip(50, 100)

# Train model
X_train = train_df.drop("potency_percent", axis=1)
y_train = train_df["potency_percent"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Align input columns with model
input_data = input_data[X_train.columns]

# Predict potency
predicted_potency = model.predict(input_data)[0]

# Output
st.subheader("📊 Predicted Potency")
st.metric("Potency (%)", f"{predicted_potency:.2f}")
