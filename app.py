import streamlit as st
import pickle

model = pickle.load(open('Pickel Files','rb'))

st.title("Brain Tumor Detection")
st.markdown("Here we are using zernike moments extracted from the tumor images")

st.subheader("Enter the Zernike moment")
deg = st.text_input('', 0,100)

st.subheader("Predicted Revenue")
st.code(float(model.predict([[deg]])))
