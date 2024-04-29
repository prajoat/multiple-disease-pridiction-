import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import math
import joblib
from PIL import Image


# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = joblib.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))




# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['\U0001F3E0 Home',
                            'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Feedback & Contact',
                            'About Us'],
                           menu_icon='hospital-fill',
                           icons=['\U0001F3E0 Home', 'activity', 'heart', 'person', 'person', 'envelope-fill'],
                           default_index=0)


if selected == '\U0001F3E0 Home':
    # Page title
    st.title('Multiple Disease Prediction')

    # Add the paragraph without alignment style
    st.markdown("This Web Application is designed to help users predict the likelihood of developing certain diseases based on their input features. With the use of trained and tested machine learning models, we provide predictions for Diabetes, Heart Disease, and Lung Cancer.")
    st.title('How To Use:')
    
    # Display instructions
    instructions = """
    1. Navigate to the Main Menu(>) located in the top-left corner of the screen.
    2. Click on the desired tab among 'Diabetes Prediction', 'Heart Disease', 'Lung Cancer' etc to access prediction tools for specific diseases.
    3. Enter relevant information as requested in the input fields.
    4. Click on the "Test Result" button to obtain predictions based on the provided data.
    """
    st.markdown(instructions)
    st.title('Disclaimer:')
    additional_info = """
    1. This Web App may not provide accurate predictions at all times. When in doubt, please enter the values again and verify the predictions.
    
    2. You are requested to provide your Name and Email for sending details about your test results. Rest assured, your information is safe and will be kept confidential.
    
    3. It is important to note that individuals with specific risk factors or concerns should consult with healthcare professionals for personalized advice and management.
    """
    st.markdown(additional_info)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex = 0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1

    with col2:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col3:
        Glucose = st.text_input('Glucose Level')

    with col1:
        BloodPressure = st.text_input('Blood Pressure value')

    with col2:
        SkinThickness = st.text_input('Skin Thickness value')

    with col3:
        Insulin = st.text_input('Insulin Level')

    with col1:
        BMI = st.text_input('BMI value')

    with col2:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col3:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

    # Button for About Diabetes
    if st.button('About Diabetes'):
            st.title('All You Need to Know About Diabetes')
            st.write("""
            Diabetes is a chronic condition that affects how your body turns food into energy.
            There are three main types of diabetes: type 1, type 2, and gestational diabetes.
            Type 1 diabetes occurs when your immune system attacks and destroys the insulin-producing cells in your pancreas.
            Type 2 diabetes occurs when your body doesn't use insulin properly or when the pancreas can't make enough insulin.
            Gestational diabetes develops during pregnancy and may go away after giving birth.
            """)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease prediction")
    
    # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
    # columns
    # no inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
    with col2:
        sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            sex = 1
        elif value == "female":
            sex = 0
    with col3:
        cp=0
        display = ("typical angina","atypical angina","non — anginal pain","asymptotic")
        options = list(range(len(display)))
        value = st.selectbox("Chest_Pain Type", options, format_func=lambda x: display[x])
        if value == "typical angina":
            cp = 0
        elif value == "atypical angina":
            cp = 1
        elif value == "non — anginal pain":
            cp = 2
        elif value == "asymptotic":
            cp = 3
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")

    with col2:
        chol = st.text_input("Serum Cholestrol")
    
    with col3:
        restecg=0
        display = ("normal","having ST-T wave abnormality","left ventricular hyperthrophy")
        options = list(range(len(display)))
        value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
        if value == "normal":
            restecg = 0
        elif value == "having ST-T wave abnormality":
            restecg = 1
        elif value == "left ventricular hyperthrophy":
            restecg = 2

    with col1:
        exang=0
        thalach = st.text_input("Max Heart Rate Achieved")
   
    with col2:
        oldpeak = st.text_input("ST depression induced by exercise relative to rest")
    with col3:
        slope=0
        display = ("upsloping","flat","downsloping")
        options = list(range(len(display)))
        value = st.selectbox("Peak exercise ST segment", options, format_func=lambda x: display[x])
        if value == "upsloping":
            slope = 0
        elif value == "flat":
            slope = 1
        elif value == "downsloping":
            slope = 2
    with col1:
        ca = st.text_input("Number of major vessels (0–3) colored by flourosopy")
    with col2:
        thal=0
        display = ("normal","fixed defect","reversible defect")
        options = list(range(len(display)))
        value = st.selectbox("thalassemia", options, format_func=lambda x: display[x])
        if value == "normal":
            thal = 0
        elif value == "fixed defect":
            thal = 1
        elif value == "reversible defect":
            thal = 2
    with col3:
        agree = st.checkbox('Exercise induced angina')
        if agree:
            exang = 1
        else:
            exang=0
    with col1:
        agree1 = st.checkbox('fasting blood sugar > 120mg/dl')
        if agree1:
            fbs = 1
        else:
            fbs=0
    # code for prediction
    heart_dig = ''
    

    # button
    if st.button("Heart test result"):
        heart_prediction=[[]]
        # change the parameters according to the model
        
        # b=np.array(a, dtype=float)
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'we are really sorry to say but it seems like you have Heart Disease.'
            
            
        else:
            heart_dig = "Congratulation , You don't have Heart Disease."
            
        st.success(name +' , ' + heart_dig)


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)



import streamlit as st

# Function to display stars based on rating
def display_stars(rating):
    stars = ""
    for i in range(5):
        if i < rating:
            stars += "★"
        else:
            stars += "☆"
    return stars

if selected == 'Feedback & Contact':
    # Page title
    st.title('Feedback & Contact')

    # Feedback form
    st.write("We'd love to hear from you! If you have any feedback, questions, or suggestions, please don't hesitate to reach out to us.")
    feedback = st.text_area("Please share your feedback here:")

    # Star rating
    st.subheader("Rate Your Experience (out of 5 stars):")
    rating = st.radio("", options=[1, 2, 3, 4, 5], format_func=display_stars)

    # Submit button
    submitted = st.button("Submit Feedback")
    if submitted:
        if feedback.strip() != "":
            # Here you can add code to handle the feedback submission, such as storing it in a database
            st.success("Thank you for your feedback! We'll review it.")
        else:
            st.warning("Please provide your feedback before submitting.")

    # Contact information
    st.subheader("Contact Information")
    st.write("You can also contact us via email or phone:")
    st.write("- Email: contact@example.com")
    st.write("- Phone: +1234567890")

if selected == 'About Us':
    st.title('Discover Our Team')

    st.write("As a team of passionate individuals, we embarked on a journey to create a user-friendly and efficient application to predict diseases such as Diabetes, Heart Disease, and Lung Cancer.")

    st.subheader("Team Members:")
    st.write("• Manthan Ninawe")
    st.write("• Tushar Singh")
    st.write("• Pajoat Padole")
    
    st.subheader("Guide: •  Prof.Dilip Motwani")
   
    

    st.write("Throughout the development process, we have combined our diverse skills and knowledge to deliver a robust and accurate disease prediction system. We are committed to promoting health awareness and providing a valuable tool for individuals to assess their health risks.")

    st.write("Thank you for choosing our Multiple Disease Prediction Web App. We hope it proves to be a valuable resource.")

