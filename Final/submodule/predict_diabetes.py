import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.neighbors import KNeighborsClassifier

# Load your pre-trained model (make sure it's saved as a .pkl file)
model = KNeighborsClassifier()  # Placeholder, load your actual trained model
scaler = MinMaxScaler()
smo = SMOTEENN(random_state=42)

def predict_diabetes(input_data):
    # Scale the input data
    scaled_data = scaler.transform(np.array(input_data).reshape(1, -1))
    
    # Resample the data (if needed, depending on your workflow)
    x_resampled, _ = smo.fit_resample(scaled_data, [0])  # Dummy y to fit
    prediction = model.predict(x_resampled)

    return prediction[0]
    
joblib.dump(predict_diabetes, "predict_diabetes.pkl")
