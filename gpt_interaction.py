import openai
import streamlit as st
def get_diagnosis_explanation(name, dataToPredic):
    api_key = st.secrets['OPENAI_API_KEY']
    openai.api_key = api_key

    prompt = f"""You are a specialized Heart Disease Doctor. Explain to the {name} their disease due to which they have heart disease in layman's terms.
    Give some examples of their disease so that they easily understand.
    Recommend some exercises to help them feel better. {name} details: {dataToPredic}.
    Give response within 340 words not more than that.
    In last write the recovery quotes for the {name} """
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
