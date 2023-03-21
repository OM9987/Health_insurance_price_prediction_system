import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

pipe_dtc = pickle.load(open('pipe2.pkl','rb'))
pipe_rf = pickle.load(open('pipe_rf.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Medical Insurance Price Predictor")

Age = st.number_input('Age of the Person')

Gender = st.selectbox('Gender' , df['sex'].unique())

bmi = st.number_input('Enter the BMI of the person')

list_child = df['children'].unique().tolist()
list_child.sort()

children = st.selectbox('no of children'  ,list_child )

smoker = st.selectbox("Do the person smoke"  , df['smoker'].unique())

region = st.selectbox("Region" , df['region'].unique())



if st.button('Predict Price'):
    query = np.array([Age, Gender, bmi, children, smoker, region])

    query = query.reshape(1, 6)
    st.title("The predicted price  is " + str(int(pipe_dtc.predict(pd.DataFrame(columns=['age','sex','bmi','children','smoker','region'],data=query))[0])))