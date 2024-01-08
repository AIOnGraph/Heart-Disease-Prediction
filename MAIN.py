import streamlit as st
import joblib
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import base64
from gpt import get_diagnosis_explanation
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
import time
@st.cache_data
def load_models():
    models = {
    "Decision Tree": {
        "model": joblib.load("decision_tree_model.pkl"),
        "scaler": joblib.load('scaler.pkl'),
    },
    "Logistic Regression": {
        "model": joblib.load("Logistic_regression_model.pkl"),
        "scaler": joblib.load('scaler.pkl'),
    },
    "KNN": {
        "model": joblib.load("KNN_model.pkl"),
        "scaler": joblib.load('scaler.pkl'),
    },
    "Naive Bayes": {
        "model": joblib.load("naive_bayes_model.pkl"),
        "scaler": joblib.load('scaler.pkl'),
    },
    "SVM": {
        "model": joblib.load("svc_model.pkl"),
        "scaler": joblib.load('scaler.pkl'),
    },
    "Deep Learning": {
        "model": joblib.load("deep_learning_model.pkl"),
        "scaler": joblib.load('scaler.pkl'),
    },
}
    return models

st.set_page_config(
    layout="wide",
    page_title="Heart Disease Prediction",
    page_icon=":heart:",
)
with st.container():
    side_bg_ext = 'png'
    side_bg = 'heartimage.png'
    st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
background-size: 200%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
  </style>
    """,
    unsafe_allow_html=True,
    )

models = load_models()

@st.cache_data
def get_prediction(dataToPredic):
    positive_predictions = 0
    predicted_models = []

    for model_name, model_dict in models.items():
        user_data_scaled = model_dict["scaler"].transform(dataToPredic)
        prediction = model_dict["model"].predict(user_data_scaled)[0]

        if prediction == 1:
            positive_predictions += 1
            predicted_models.append(model_name)

    threshold = 3

    if positive_predictions >= threshold:
        final_prediction = 1
    else:
        final_prediction = 0

    return final_prediction, predicted_models



st.markdown("<h1 style='text-align: center; color: black;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)

st.write(
    "Welcome to the Heart Disease Prediction App! 🌟\n"
    "Let's embark on a journey to explore your heart health. Fill in the details below and let the magic happen!"
)


st.subheader('Tell us more about yourself:')
col1,col2 = st.columns(2)
with col1:
    name=st.text_input('**Name**')
    age = st.slider('**Age**', min_value=18, max_value=100, step=1)
    sex=st.selectbox('**Gender**',options=['Female','Male'])
    if sex=="Male":
        sex=1
    else:
        sex=0
        
    cp=st.selectbox('**Chest Pain**',options=['None','Typical Angina','Atypical Angina','Non-Angina','Asymptomatic'])
    if cp=='Typical Angina':
        cp=1
    elif cp=='Atypical Angina':
        cp=2
    elif cp== 'Non-Angina':
        cp=3
    elif cp=='Asymptomatic':
        cp=4
    else:
        cp=0
    trestbps = st.slider('**BP**', min_value=50, max_value=250, step=1)  
    chol = st.slider('**Cholesterol**', min_value=100, max_value=500, step=1)  
    fbs=st.selectbox('**Blood Sugar**',options=['> 120 mg/dl','< 120 mg/dl'])
    if fbs=='> 120 mg/dl':
        fbs=1
    else:
        fbs=0
with  col2: 
    restecg=st.selectbox('**ECG**',options=['Normal','Having ST-T wave abnormality','hypertrophy'])
    if restecg=='Having ST-T wave abnormality':
        restecg=1
    elif restecg=='hypertrophy':
        restecg=2
    else:
        restecg=0
    thalach = st.slider('**Heart Rate**', min_value=50, max_value=250, step=1)
    exang=st.selectbox('**Exercise**',options=['No','Yes'])
    if exang=='Yes':
        exang=1
    else:
        exang=0
    oldpeak = st.slider('**Depression**', min_value=0.0, max_value=5.0, step=0.1)
    slope=st.selectbox('**Slope**',options=['Upsloping','Flat','Down Sloping'])
    if slope=='Upsloping':
        slope=1
    elif slope=='Flat':
        slope=2
    else:
        slope=3
    ca=st.selectbox('**Vessels**',options=['None','One','Two','Three'])
    if ca=='One':
        ca=1
    elif ca=='Two':
        ca=2
    elif ca=='Three':
        ca=3
    else:
        ca=0
    thal=st.selectbox('**Scan**',options=['Normal','Fixed Defect','Reversible Defect'])
    if thal=='Normal':
        thal=3
    elif thal=='Fixed Defect':
        thal=6
    else:
        thal=7

   
dataToPredic = pd.DataFrame({
"age": [age],
"sex": [sex],
"cp": [cp],
"trestbps": [trestbps],
"chol": [chol],
"fbs": [fbs],
"restecg": [restecg],
"thalach": [thalach],
"exang": [exang],
"oldpeak": [oldpeak],
"slope": [slope],
"ca": [ca],
"thal": [thal],
 })

openai_key = None
predict_button = st.button('Predict My Heart Health', key="predict_button")

if predict_button:
    with st.spinner('Analyzing...'):
        final_prediction, predicted_models = get_prediction(dataToPredic)
        time.sleep(3)

    st.success('Analysis Complete!')

    if final_prediction == 1:
        st.warning('Warning: It seems there might be an issue with your heart health. Consult with a healthcare professional.')
    else:
        st.success('Great news! Your heart health appears to be in good condition.')

more_info_checked = st.checkbox("Do you want more information regarding this disease", value=False)

if more_info_checked:
    openai_key = st.text_input('Enter your OpenAI key:', type="password", key="openai_key_input")

    if openai_key:
        
        chat = ChatOpenAI(streaming=True,
                      callbacks=[StreamlitCallbackHandler(st.empty())],
                        verbose=True, 
                        temperature=0,
                        api_key=openai_key
                        )
        
        button = st.button('Submit',on_click=get_diagnosis_explanation,args=[name, dataToPredic, openai_key,chat])
        if button:
            st.success('Done!')
            