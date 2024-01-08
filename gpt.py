import openai
import streamlit as st
import os
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
from promt import Prompt
def get_diagnosis_explanation(name, dataToPredic, openai_key,chat):
    openai.api_key = openai_key

    prompt = f"{Prompt},Patient:{name},dataToPredict:{dataToPredic}"

    messages = [
        SystemMessage(content=prompt),
    ]
    
    st.session_state = chat(messages)