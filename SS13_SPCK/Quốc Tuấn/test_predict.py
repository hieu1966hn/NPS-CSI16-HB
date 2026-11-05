import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid Metal GPU hangs on macOS

import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import sklearn.compose._column_transformer as _ct

if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

print("Loading model and preprocessor...")
model = tf.keras.models.load_model("student_score_model.keras")
preprocessor = joblib.load("preprocessor.pkl")

# Mock data matching the columns of the dataset
student_df = pd.DataFrame({
    "Hours_Studied": [15],
    "Attendance": [85],
    "Parental_Involvement": ["Low"],
    "Access_to_Resources": ["Low"],
    "Extracurricular_Activities": ["Yes"],
    "Sleep_Hours": [7],
    "Previous_Scores": [75],
    "Motivation_Level": ["Low"],
    "Internet_Access": ["Yes"],
    "Tutoring_Sessions": [2],
    "Family_Income": ["Low"],
    "Teacher_Quality": ["Low"],
    "School_Type": ["Public"],
    "Peer_Influence": ["Positive"],
    "Physical_Activity": [5],
    "Learning_Disabilities": ["Yes"],
    "Parental_Education_Level": ["High School"],
    "Distance_from_Home": ["Near"],
    "Gender": ["Male"]
})

print("Transforming input...")
try:
    X = preprocessor.transform(student_df)
    print("X shape:", X.shape)
    print("Predicting...")
    prediction = model.predict(X)
    print("Prediction:", prediction)
except Exception as e:
    print("Error during transformation/prediction:", e)
