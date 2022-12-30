import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

st.write("""
# Wine Class Prediction App
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    Alcohol = st.sidebar.slider('Alcohol', 11.0, 14.8)
    Malic_Acid = st.sidebar.slider('Malic_Acid', 0.74, 5.80)
    Ash = st.sidebar.slider('Ash', 1.36, 3.23)
    Alcalinity_of_Ash = st.sidebar.slider('Alcalinity_of_Ash', 10.6, 30.0)
    Magnesium = st.sidebar.slider('Magnesium', 70.0, 162.0)
    Total_Phenols = st.sidebar.slider('Total_Phenols', 0.98, 3.88)
    Flavanoids = st.sidebar.slider('Flavanoids', 0.34, 5.08)
    Nonflavanoid_Phenols = st.sidebar.slider('Nonflavanoid_Phenols', 0.13, 0.66)
    Proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58)
    Colour_Intensity = st.sidebar.slider('Colour_Intensity', 1.3, 13.0)
    Hue = st.sidebar.slider('Hue', 0.48, 1.71)
    protein_content_of_various_wines = st.sidebar.slider('protein_content_of_various_wines', 1.27, 4.0)
    Proline = st.sidebar.slider('Proline', 278.0, 1680.0)
    data = {'Alcohol': Alcohol,
            'Malic_Acid': Malic_Acid,
            'Ash': Ash,
            'Alcalinity_of_Ash': Alcalinity_of_Ash,
            'Magnesium': Magnesium,
            'Total_Phenols': Total_Phenols,
            'Flavanoids': Flavanoids,
            'Nonflavanoid_Phenols': Nonflavanoid_Phenols,
            'Proanthocyanins': Proanthocyanins,
            'Colour_Intensity': Colour_Intensity,
            'Hue': Hue,
            'protein_content_of_various_wines': protein_content_of_various_wines,
            'Proline': Proline}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

wine = datasets.load_wine()
X = wine.data
y = wine.target

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labes and their corresponding index number')
st.write(wine.target_names)

st.subheader('Prediction')
st.write(wine.target_names[prediction])
#st.write(predicion)

st.subheader('Prediction Probability')
st.write(prediction_proba)
            