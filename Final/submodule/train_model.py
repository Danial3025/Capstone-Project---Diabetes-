import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
import joblib

def train_model(data):
    # Prepare X and y
    x = data.drop(columns='Outcome', axis=1)
    y = data['Outcome']

    # Scale the features
    scale = MinMaxScaler()
    x = scale.fit_transform(x)

    # Apply SMOTEENN
    smoten = SMOTEENN(random_state=42)
    x_resampled, y_resampled = smoten.fit_resample(x, y)

    # Split the dataset into training and testing sets
    x_train_resampled, x_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
        x_resampled, y_resampled, test_size=0.2, random_state=42, shuffle=True, stratify=y_resampled
    )

    # Train the KNN model
    model = KNeighborsClassifier()
    model.fit(x_train_resampled, y_train_resampled)

    # Cross-validation scores
    cv_scores = cross_val_score(model, x_train_resampled, y_train_resampled, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)

    # Calculate test accuracy
    test_accuracy = model.score(x_test_resampled, y_test_resampled)

    # Save the trained model and scaler
    joblib.dump(model, "submodule/knn_model.pkl")
    joblib.dump(scale, "submodule/scaler.pkl")

    return cv_scores.mean(), test_accuracy
    
    # Save the trained model
joblib.dump(train_model, "train_model.pkl")


# Note: You may call this function in your main dashboard to train the model.
