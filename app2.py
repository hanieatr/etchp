import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from PIL import Image
import time

# Title of the app
st.title("Solar Thermal Collector Performance Prediction")

# Load and display image
photo = Image.open('gui2.png')
st.image(photo)
# Dictionary to map fluid names to saturation temperatures
fluid_temps = {
    'Acetone': 56.24,
    'Methanol': 65,
    'Ethanol': 78.4,
    'Water': 100
}
# Sidebar inputs
manifold_fluid = st.sidebar.radio('Manifold fluid', ['Water', 'Air'])
#using dct
fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
T_sat = fluid_temps[fluid_name] # Get saturation temperature from dictionary

N = st.sidebar.number_input('Number of tubes (N)', min_value=1)
D_o = st.sidebar.number_input('Outer diameter of tube (D_o)')
L_tube = st.sidebar.number_input('Length of the tube (L_tube)')
L_c = st.sidebar.number_input('Length of the condenser (L_c)')
L_e = st.sidebar.number_input('Length of the evaporator (L_e)')
L_a = st.sidebar.number_input('Length of the adiabatic section (L_a)')
De_o = st.sidebar.number_input('Outer diameter of evaporator (De_o)')
t_e = st.sidebar.number_input('Thickness of evaporator (t_e)')
Dc_o = st.sidebar.number_input('Outer diameter of condenser (Dc_o)')
t_c = st.sidebar.number_input('Thickness of condenser (t_c)')
theta = st.sidebar.number_input('Angle of radiation (theta)')
t_g = st.sidebar.number_input('Thickness of glass (t_g)')
D_H = st.sidebar.number_input('Hydraulic diameter of tube (D_H)')
A_man = st.sidebar.number_input('Area of manifold (A_man)')
alpha_ab = st.sidebar.number_input('Absorptivity of absorber (alpha_ab)')
epsilon_ab = st.sidebar.number_input('Emissivity of absorber (epsilon_ab)')
epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)')
tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)')
I = st.sidebar.number_input('Solar irradiance (I)')
T_amb = st.sidebar.number_input('Ambient temperature (T_amb)')
U_amb = st.sidebar.number_input('Velocity of wind (U_amb)')
T_in = st.sidebar.number_input('Inlet temperature (T_in)')
m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)')

# Prepare input data for prediction
input_data = pd.DataFrame({
    'T_sat': [T_sat],
    'N': [N],
    'D_o': [D_o],
    'L_tube': [L_tube],
    'L_c': [L_c],
    'L_e': [L_e],
    'L_a': [L_a],
    'De_o': [De_o],
    't_e': [t_e],
    'Dc_o': [Dc_o],
    't_c': [t_c],
    'theta': [theta],
    't_g': [t_g],
    'D_H': [D_H],
    'A_man': [A_man],
    'alpha_ab': [alpha_ab],
    'epsilon_ab': [epsilon_ab],
    'epsilon_g': [epsilon_g],
    'tau_g': [tau_g],
    'I': [I],
    'T_amb': [T_amb],
    'U_amb': [U_amb],
    'T_in': [T_in],
    'm_dot': [m_dot]
})
#.................create process bar
progress_bar=st.progress(0)

# Load data based on fluid type selection
if manifold_fluid == 'Water':
    # Load models and scalers for Water
    loaded_model_eta = xgb.XGBRegressor(n_estimators=5000)
    loaded_model_eta.load_model('xgbwatereta_model.json')
    loaded_model_T = xgb.XGBRegressor(n_estimators=5000)
    loaded_model_T.load_model('xgbwaterT_model.json')

    with open('wateretax.pkl', 'rb') as f:
        x_scaler_eta = pickle.load(f)
    with open('wateretay.pkl', 'rb') as f:
        scaler_y_eta = pickle.load(f)

    with open('waterTx.pkl', 'rb') as f:
        x_scaler_T = pickle.load(f)
    with open('waterTy.pkl', 'rb') as f:
        scaler_y_T = pickle.load(f)
        #............................
    progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models
    input_data_scaled_eta = x_scaler_eta.transform(input_data.values.reshape(1, -1))
    y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
    y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))

    input_data_scaled_T = x_scaler_T.transform(input_data.values.reshape(1, -1))
    y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
    y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
    progress_bar.progress(66)
    
    # Display prediction results
    st.subheader("Predicted Outputs for Water:")
    st.write("Predicted Efficiency (eta):", y_pred_eta[0][0])
    st.write("Predicted Temperature (T):", y_pred_T[0][0]) 

    progress_bar.progress(100)
else: # manifold_fluid == 'Air'
    loaded_model_eta = xgb.XGBRegressor(n_estimators=5000)
    loaded_model_eta.load_model('xgbaireta_model.json')
    loaded_model_T = xgb.XGBRegressor(n_estimators=5000)
    loaded_model_T.load_model('xgbairrT_model.json')

    with open('airetax.pkl', 'rb') as f:
        x_scaler_eta = pickle.load(f)
    with open('airetay.pkl', 'rb') as f:
        scaler_y_eta = pickle.load(f)

    with open('airTx.pkl', 'rb') as f:
        x_scaler_T = pickle.load(f)
    with open('airTy.pkl', 'rb') as f:
        scaler_y_T = pickle.load(f)
        #..............
    progress_bar.progress(33) 

    # Make predictions for both models
    input_data_scaled_eta = x_scaler_eta.transform(input_data.values.reshape(1, -1))
    y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
    y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))

    input_data_scaled_T = x_scaler_T.transform(input_data.values.reshape(1, -1))
    y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
    y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
    #......
    progress_bar.progress(66) 

    # Display prediction results
    st.subheader("Predicted Outputs for Water:")
    st.write("Predicted Efficiency (eta):", y_pred_eta[0][0])
    st.write("Predicted Exit Temperature (T_exit):", y_pred_T[0][0])
    progress_bar.progress(100) 