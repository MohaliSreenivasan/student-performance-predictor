print("Student Performance Predictor(ML Project)")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'hours_studied': [1,2,3,4,5,6,7,8,9,10],
    'attendance': [50,55,60,65,70,75,80,85,90,95],
    'marks': [30,35,40,45,50,55,60,65,70,75]
}

# ✅ CREATE df FIRST
df = pd.DataFrame(data)

# Features & target
X = df[['hours_studied', 'attendance']]
y = df['marks']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ User input
hours = float(input("Enter hours studied: "))
attendance = float(input("Enter attendance: "))

# Prediction
import pandas as pd
input_data = pd.DataFrame([[hours, attendance]], columns=['hours_studied', 'attendance'])
prediction = model.predict(input_data)

print("Predicted Marks:", prediction[0])