import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import math

#Main title and heading
st.write("""
# Diabetes Detection
Detect the presence of diabetes
""")

image = Image.open('image.jpg')

#Banner image
st.image(image, use_column_width=True)

data_df = pd.read_csv('diabetes.csv')

#Create a subheader
st.subheader('Data Complete Information')

#Show data as a table
st.dataframe(data_df)

#show some statistics
st.write(data_df.describe())

#show data as a chart
chart = st.bar_chart(data_df)

#split data into X and Y
X = data_df.iloc[:, 0:8].values
Y = data_df.iloc[:, -1].values

#split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

#Get all user input
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    blood_glucose = st.sidebar.slider('blood_glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)
        
    #store all input
    user_inputs = {
        'pregnancies': pregnancies,
        'blood_glucose': blood_glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'Age': Age
    }
           
    #Transform data into a data frame
    features = pd.DataFrame(user_inputs, index=[0])
    return features

#Store user input into a variable
input_data = get_user_input()

st.subheader("User Input: ")
st.write(input_data)

#Create and train model
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, Y_train)

#Show model metrics
st.subheader("Model Test Accuracy Score: ")
st.write(str(math.floor(accuracy_score(Y_test, randomForestClassifier.predict(X_test)) * 100)) + "%")
st.subheader("Model Confusion Matrix: ")
st.write(confusion_matrix(Y_test, randomForestClassifier.predict(X_test)))


#Store model predictions
prediction = randomForestClassifier.predict(input_data)

st.subheader('Classification: ')
st.write(prediction)