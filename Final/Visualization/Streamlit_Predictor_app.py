import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTEENN
from submodule.load_data import load_data
from submodule.plot_graph import plot_graph
from submodule.train_model import train_model

def train_and_predict(data, input_data):
    """Train the KNN model and predict diabetes based on input data."""
    # Prepare X and y
    x = data.drop(columns='Outcome', axis=1)
    y = data['Outcome']

    # Scale the features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    # Apply SMOTEENN
    smoten = SMOTEENN(random_state=42)
    x_resampled, y_resampled = smoten.fit_resample(x_scaled, y)

    # Train the KNN model
    model = KNeighborsClassifier()
    model.fit(x_resampled, y_resampled)

    # Scale the input data for prediction
    input_scaled = scaler.transform([input_data])

    # Make the prediction
    prediction = model.predict(input_scaled)

    return prediction[0]

def display_info_page():
    """Display images and text on the new page."""
    st.title("Diabetes")
    
    # Display a sentence
    st.write("Welcome to the Diabetes Awareness Page!")
    
    # Display an image
    st.image("application/img/dbts.jpg", caption="Understanding Diabetes", use_column_width=True)
    
    # Add more text
    st.write("What is diabetes? ")

    st.write("Diabetes occurs when your body does not process food as energy properly. Insulin is a critical hormone that gets glucose (sugar that is used as energy) to the cells in your body. When you have diabetes, your body either doesn’t respond to insulin or doesn’t produce insulin at all. This causes sugars to build up in your blood, which puts you at risk of dangerous complications..")
    
    # Display another image
    st.image("application/img/symptoms2.jpg", caption="Diabetes Symptoms", use_column_width=True)
    
    # Display another image
    st.image("application/img/check.jpg", caption="Glucometer", use_column_width=True)
    
    st.markdown("""
    A doctor can ***diagnose diabetes*** with one or more of the following blood tests:
    - Random blood sugar test: Taken any time, regardless of how recently you have eaten.
    - Fasting blood sugar test: Measures blood sugar levels after you have not eaten overnight.
    - Glucose tolerance test: Takes blood levels over the course of several hours to show how quickly your body metabolizes the glucose in a special liquid you drink. """)
    
def main():
    st.title("Streamlit Dashboard")
    data = load_data()
    # Set the default selection to "Information"
    page = st.sidebar.selectbox("Select a page", ["Information", "Homepage", "Exploration", "Modelling", "Prediction"])

    if page == "Information":
        display_info_page()

    elif page == "Homepage":
        st.title("Homepage")
        st.text("Diabetes Dataset")
        st.dataframe(data)

        st.title("Distribution of Target Classes")
        
        # Create the countplot
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Outcome', data=data, palette='pastel')
        plt.title('Distribution of Target Classes')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after displaying

        # Print value counts in Streamlit
        st.write(data['Outcome'].value_counts())

    elif page == "Exploration":
        st.title("Exploratory Data Analysis")
        st.text("Correlation Coefficient")
        
        fig1, fig2, fig3 = plot_graph(data)  # Expecting three figures
        st.pyplot(fig1)  # Display heatmap
        st.pyplot(fig2)  # Display pairplot
        st.plotly_chart(fig3)  # Display interactive histogram

    elif page == "Modelling":
        st.title("Model Training")
        
        # Train the KNN model and display results
        if st.button("Train KNN Model"):
            mean_accuracy, test_accuracy = train_model(data)
            st.write(f"Mean Cross-Validation Accuracy for KNN: {mean_accuracy:.4f}")
            st.write(f"Test Accuracy for KNN: {test_accuracy:.4f}")

    elif page == "Prediction":
        st.title("Predict Diabetes")
        st.write("Enter the following details to predict diabetes:")

        # Input fields for prediction using sliders
        pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.slider("Glucose", min_value=0, max_value=200, step=1)
        blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=200, step=1)
        skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=100, step=1)
        insulin = st.slider("Insulin", min_value=0, max_value=1000, step=1)
        bmi = st.slider("BMI", min_value=0.0, max_value=50.0, step=0.1)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
        age = st.slider("Age", min_value=0, max_value=120, step=1)

        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

        if st.button("Predict"):
            result = train_and_predict(data, input_data)
            if result == 1:
                st.success("Predicted Outcome: Diabetes")
                st.progress(1.0)
            else:
                st.success("Predicted Outcome: No Diabetes")
                st.progress(0.0)

if __name__ == "__main__":
    main()
