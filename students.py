# AI model for predicting student performance based on study hours and class attendance using logistic regression.
#  The model is trained on a dataset of students' study hours, class attendance, and their pass/fail status.
#  The user can input their study hours and class attendance to get a prediction of their probability of passing. 
# The model's accuracy is also evaluated on a test set.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(study, passed, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression()

model.fit(X_train, y_train)
# Predicting the performance of students based on study hours
user_study_hours = int(input("Enter the number of study hours: "))
user_class_attendance = int(input("Enter the class attendance: "))
if user_study_hours < 0 or user_class_attendance < 0:
    print("Study hours and class attendance cannot be negative.")
    exit()

new_data = pd.DataFrame({
    'StudyHours': [user_study_hours],
    'class_attendance': [user_class_attendance]
})

# Scaling the new data using the same scaler used for training
new_data = scaler.transform(new_data)
probability = model.predict_proba(new_data)[:, 1][0]
probability = round(probability * 100, 2) # Converting to percentage and rounding to 2 decimal places

# Displaying the predictions
print("Probability of passing:", probability, "%")

# Evaluating the model weights on the features
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

#Making predictions on the test set
y_pred = model.predict(X_test)
# Evaluating the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
