import pickle
import streamlit as st
import pandas as pd
import base64

# Load the pre-trained model
model = pickle.load(open("predictor.pkl", "rb"))

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    sex = 1 if sex == 'male' else 0
    embarked_map = {'C': 0, 'Q': 1, 'S': 2}
    embarked = embarked_map[embarked]
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    return input_data

st.markdown("<h1 style='text-align: center; color: #FFFFFF; font-family: Arial; font-size: 50px;'>Titanic Survival Prediction</h1>", unsafe_allow_html=True)

def titanic_prediction():
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">Pclass</span>', unsafe_allow_html=True)
    pclass = st.selectbox("", [1, 2, 3], index=2, key="pclass")
    
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">Sex</span>', unsafe_allow_html=True)
    sex = st.selectbox("", ['male', 'female'], index=0, key="sex")
    
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">Age</span>', unsafe_allow_html=True)
    age = st.number_input("", min_value=0, max_value=100, value=25, key="age")
    
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">SibSp (Number of Siblings/Spouses Aboard)</span>', unsafe_allow_html=True)
    sibsp = st.number_input("", min_value=0, max_value=10, value=0, key="sibsp")
    
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">Parch (Number of Parents/Children Aboard)</span>', unsafe_allow_html=True)
    parch = st.number_input("", min_value=0, max_value=10, value=0, key="parch")
    
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">Fare</span>', unsafe_allow_html=True)
    fare = st.number_input("", min_value=0.0, value=50.0, key="fare")
    
    st.markdown('<span style="color:white; font-size: 30px; font-family: Arial;">Embarked</span>', unsafe_allow_html=True)
    embarked = st.selectbox("", ['C', 'Q', 'S'], index=2, key="embarked")
    
    pred = st.button("Predict")

    if pred:
        input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write('<span style="color:white; font-size: 40px; font-family: Arial;">The passenger would have survived.</span>', unsafe_allow_html=True)
        else:
            st.write('<span style="color:white; font-size: 40px; font-family: Arial;">The passenger would not have survived.</span>', unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    add_bg_from_local('titanic.jpg')
    titanic_prediction()
