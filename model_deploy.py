import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# Function to set the background image
def set_background(url, width="84%", height="109", position="right"):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{url}");
            background-size: {width} {height};
            background-position: {position};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app with sleep wellness theme background
st.set_page_config(page_title="Sleep Analysis", page_icon="")

# Function to load model and scaler based on user selection
@st.cache_data
def load_assets(selected_model):
    model_filename = f"{selected_model}.pkl"
    with open(model_filename, 'rb') as f:
        best_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return best_model, scaler

# User selection for model
model_options = {
    "Random Forest Regression": "randomforestregressor_model",
    "Histogram Gradient Boosting": "histgradientboostingregressor_model",
}
selected_model = st.sidebar.selectbox("Choose Prediction Model", list(model_options.keys()))

# Load assets based on selection
best_model, scaler = load_assets(model_options[selected_model])

# Set the background image
set_background('https://my-wall-clock.com/cdn/shop/articles/Leonardo_Diffusion_XL_A_bedroom_with_an_alarm_clock_that_emit_0_1344x.jpg?v=1706073439', width="84%", height="109%", position='right')

# Streamlit app title
title_container = st.container()
with title_container:
    st.title('Sleep Well, Achieve More: Sleep Analysis Model')

# Adjust layout for mobile screens
st.markdown("""
<style>
@media only screen and (max-width: 768px) {
    .stApp {
        flex-direction: column;
    }
    #title {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# User input
st.sidebar.header('Track Your Sleep Habits')
total_sleep_time = st.sidebar.slider('Total Sleep Time (hours)', min_value=0, max_value=24, value=8)
midpoint_sleep = st.sidebar.slider('Midpoint of Sleep (hour)', min_value=0, max_value=24, value=12)
daytime_sleep = st.sidebar.slider('Daytime Sleep (hours)', min_value=0, max_value=10, value=1)
term_gpa = st.sidebar.slider('Term GPA', min_value=0.0, max_value=4.0, value=2.5)
term_units = st.sidebar.slider('Term Units Taken', min_value=0, max_value=20, value=15)
frac_nights_with_data = st.sidebar.slider('Fraction of Nights with Sleep Data', min_value=0.0, max_value=1.0, value=0.5)

# Handle categorical variables (demo_race, demo_gender)
demo_race_options = ['Underrepresented', 'Non-underrepresented']
demo_race_mapping = {'Underrepresented': 0, 'Non-underrepresented': 1}
demo_race_value = st.sidebar.selectbox('Race', demo_race_options)

demo_gender_options = ['Male', 'Female']
demo_gender_mapping = {'Male': 0, 'Female': 1}
demo_gender_value = st.sidebar.selectbox('Gender', demo_gender_options)

# Handle 'demo_firstgen' value conversion (assuming numerical mapping)
demo_firstgen_mapping = {'No': 0, 'Yes': 1}
demo_firstgen_value = demo_firstgen_mapping.get(st.sidebar.selectbox('First-Generation Student?', ('No', 'Yes')))

# Prepare user input data as DataFrame
feature_names = ['TotalSleepTime', 'midpoint_sleep', 'daytime_sleep', 'term_gpa', 'term_units',
                 'frac_nights_with_data', 'demo_firstgen', 'demo_race', 'demo_gender']

input_data = pd.DataFrame([[
    total_sleep_time,
    midpoint_sleep,
    daytime_sleep,
    term_gpa,
    term_units,
    frac_nights_with_data,
    demo_firstgen_value,
    demo_race_mapping[demo_race_value],
    demo_gender_mapping[demo_gender_value]
]], columns=feature_names)

# Ensure feature order matches the order used in fitting the scaler
scaled_input_data = scaler.transform(input_data[feature_names])

# Define the polynomial features transformer and transform the scaled input data
poly_features = PolynomialFeatures(degree=2)
poly_transformed_data = poly_features.fit_transform(scaled_input_data)

# Prediction button
if st.button('Predict'):
    # Make prediction using the loaded model
    prediction = best_model.predict(poly_transformed_data)[0]

    # Display prediction
    st.write('Predicted Cumulative GPA:')
    st.success(round(prediction, 2))
