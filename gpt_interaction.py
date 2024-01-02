import openai
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from gpt_prompt import Prompt
def get_diagnosis_explanation(name, dataToPredic, openai_key):
    openai.api_key = openai_key

    prompt = f"{Prompt},Name:{name},dataToPredict:{dataToPredic}"
    messages = [{"role": "user", "content": prompt}]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        stream=True
    )

    collected_message = ''
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            collected_message += chunk_message
            

    return collected_message
