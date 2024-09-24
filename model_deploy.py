import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor


# Function to set the background image
def set_background(url, width="100%", height="auto", position="center center"):
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
st.set_page_config(page_title="Sleep Analysis", page_icon="")  # Set page title and icon


# Load the best model and scaler
@st.cache_data
def load_assets():
    with open('histgradientboostingregressor_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:  # Load the scaler
        scaler = pickle.load(f)
    return best_model, scaler


best_model, scaler = load_assets()

# Set the background image (Use raw string to avoid unicode error)
# Set the background image with adjusted size and position
set_background('https://my-wall-clock.com/cdn/shop/articles/Leonardo_Diffusion_XL_A_bedroom_with_an_alarm_clock_that_emit_0_1344x.jpg?v=1706073439', width="84%", height="109%", position='right')  # Adjust width and height as desired

# Streamlit app title
st.title('Sleep Well, Achieve More: Sleep Analysis Model')

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
poly_features = PolynomialFeatures(degree=2)  # You can specify the degree of polynomial features required
poly_transformed_data = poly_features.fit_transform(scaled_input_data)  # Fit and transform the data

# Prediction button
if st.button('Predict'):
    # Make prediction using the loaded model
    prediction = best_model.predict(poly_transformed_data)[0]

    # Display prediction
    st.write('Predicted Cumulative GPA:')
    st.success(prediction)