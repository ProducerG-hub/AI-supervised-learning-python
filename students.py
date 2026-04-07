# AI model for preding student performance based on study hours
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Loading the dataset
data ={
    'StudyHours': [1, 2, 3, 4, 5,6,7,8,9,10],
    'class_attendance':[10, 20, 30, 40, 50,60,70,80,90,100],
    'Passed': [0, 0, 0, 0, 0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Preparing the data for training
study = df[['StudyHours','class_attendance']]
passed = df['Passed']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(study, passed, test_size=0.2)

# Training the model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predicting the performance of students based on study hours
user_study_hours = int(input("Enter the number of study hours: "))
user_class_attendance = int(input("Enter the class attendance: "))
new_data = pd.DataFrame({
    'StudyHours': [user_study_hours],
    'class_attendance': [user_class_attendance]
})
probability = model.predict_proba(new_data)[:, 1][0]
probability = round(probability * 100, 2) # Converting to percentage and rounding to 2 decimal places

# Displaying the predictions
print("Probability of passing:", probability)