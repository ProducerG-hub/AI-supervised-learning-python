# Student AI model for predicting student performance based on study hours and class attendance using logistic regression.
# Creating an API to serve the model predictions using Flask and testing on Node.js.
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
scaler = StandardScaler()

# Loading the dataset
data ={
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'class_attendance':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Passed': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Preparing the data for training
study = df[['StudyHours','class_attendance']]
passed = df['Passed']

# Scaling the features to improve model performance by balancing the features weights
study = scaler.fit_transform(study)

# Training the model
model = LogisticRegression()
model.fit(study, passed)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_study_hours = data.get('study_hours')
    user_class_attendance = data.get('class_attendance')

    if user_study_hours is None or user_class_attendance is None:
        return jsonify({'error': 'Please provide both study hours and class attendance.'}), 400

    if user_study_hours < 0 or user_class_attendance < 0:
        return jsonify({'error': 'Study hours and class attendance cannot be negative.'}), 400

    new_data = pd.DataFrame({
        'StudyHours': [user_study_hours],
        'class_attendance': [user_class_attendance]
    })

    # Scaling the new data using the same scaler used for training
    new_data = scaler.transform(new_data)
    probability = model.predict_proba(new_data)[:, 1][0]
    probability = round(probability * 100, 2) # Converting to percentage and rounding to 2 decimal places

    return jsonify({'probability_of_passing': f"{probability}% \n Thank you for using Mlue Student Performance Predictor API!"})


if __name__ == '__main__':
    app.run(debug=True)