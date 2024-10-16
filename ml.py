import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the heart disease dataset
data= pd.read_csv('heart.csv')

# Preprocess the data
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data = data[data['sex'] != 0]
data = data.reset_index(drop=True)

X = data[['age','sex','cp','trestbps','chol']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model,'model.pkl')
