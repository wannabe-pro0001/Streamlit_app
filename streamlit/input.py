from re import I
import time
import streamlit as st

st.number_input('Pick a number', 0,10)
st.text_input('Email address', placeholder="Ur email address")
st.date_input('Travelling date')
st.time_input('School time')
st.text_area('Description', placeholder="Ur fuking description")
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')

st.balloons()
st.progress(0.5)
with st.spinner('Wait for it...'):
    time.sleep(10)

st.success("You did it !")
st.error("Error")
st.warning("Warning")
st.info("It's easy to build a streamlit app")
st.exception(RuntimeError("RuntimeError exception"))

