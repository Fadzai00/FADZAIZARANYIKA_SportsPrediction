import streamlit as st
import pickle
import numpy as np


# Load model using pickle
model_path = 'best_dt_model.pkl'
st.write(f"Loading model from: {model_path}")

model = None
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded successfully.")
except FileNotFoundError as fnf_error:
    st.write(f"File not found error: {fnf_error}")
except pickle.PickleError as pickle_error:
    st.write(f"Pickle error: {pickle_error}")
except Exception as e:
    st.write(f"Error loading model: {e}")

st.title('Player Rating Prediction')

wage_eur = st.number_input('Wage (EUR)')
potential_rating = st.number_input('Potential Rating(1-100)')
movement_reactions = st.number_input('Movement Reactions(1-100)')
value_eur = st.number_input('Value (EUR)')
international_reputation = st.number_input('International Reputation(1-100)')

if st.button('Predict'):
    if model:
        try:
            features = np.array([wage_eur, potential, movement_reactions, value_eur, international_reputation]).reshape(1, -1)
            prediction = model.predict(features)[0]
            st.write(f'Prediction: {prediction}')
        except Exception as e:
            st.write(f"Error making prediction: {e}")
    else:
        st.write('Model is not loaded properly.')
