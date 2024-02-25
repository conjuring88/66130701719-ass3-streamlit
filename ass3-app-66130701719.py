import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

model = pickle.load(open('per_model-66130701719.sav', 'rb'))
st.title('Iris Flower Prediction App')

x1 = st.slider('Sepal Length', 0.0, 10.0, 0.1)
x2 = st.slider('Sepal Width', 0.0, 10.0, 0.1)
x3 = st.slider('Petal Length', 0.0, 10.0, 0.1)
x4 = st.slider('Petal Width', 0.0, 10.0, 0.1)

input_data = np.array([[x1, x2, x3, x4]]).reshape(1, -1)

if st.button('Predict'):
    result = model.predict(input_data)
    st.write('The result is:', result)
    st.write('The flower is:', result[0])