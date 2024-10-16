import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Provided data (replace this with your JSON data)
data = {
  "Fungal infection": {
    "itching": 1,
    "skin_rash": 1,
    "nodal_skin_eruptions": 1,
    "dischromic_patches": 1
  },
  "Allergy": {
    "continuous_sneezing": 1,
    "shivering": 1,
    "chills": 1,
    "watering_from_eyes": 1
  },
  "GERD": {
    "stomach_pain": 1,
    "acidity": 1,
    "ulcers_on_tongue": 1,
    "vomiting": 1,
    "cough": 1,
    "chest_pain": 1
  },
  "Chronic cholestasis": {
    "itching": 1,
    "vomiting": 1,
    "yellowish_skin": 1,
    "nausea": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "yellowing_of_eyes": 1
  },
  "Drug Reaction": {
    "itching": 1,
    "skin_rash": 1,
    "stomach_pain": 1,
    "burning_micturition": 1,
    "spotting_ urination": 1
  },
  "Peptic ulcer diseae": {
    "vomiting": 1,
    "indigestion": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "passage_of_gases": 1,
    "internal_itching": 1
  },
  "AIDS": {
    "muscle_wasting": 1,
    "patches_in_throat": 1,
    "high_fever": 1,
    "extra_marital_contacts": 1
  },
  "Diabetes ": {
    "fatigue": 1,
    "weight_loss": 1,
    "restlessness": 1,
    "lethargy": 1,
    "irregular_sugar_level": 1,
    "blurred_and_distorted_vision": 1,
    "obesity": 1,
    "excessive_hunger": 1,
    "increased_appetite": 1,
    "polyuria": 1
  },
  "Gastroenteritis": {
    "vomiting": 1,
    "sunken_eyes": 1,
    "dehydration": 1,
    "diarrhoea": 1
  },
  "Bronchial Asthma": {
    "fatigue": 1,
    "cough": 1,
    "high_fever": 1,
    "breathlessness": 1,
    "family_history": 1,
    "mucoid_sputum": 1
  },
  "Hypertension ": {
    "headache": 1,
    "chest_pain": 1,
    "dizziness": 1,
    "loss_of_balance": 1,
    "lack_of_concentration": 1
  },
  "Migraine": {
    "acidity": 1,
    "indigestion": 1,
    "headache": 1,
    "blurred_and_distorted_vision": 1,
    "excessive_hunger": 1,
    "stiff_neck": 1,
    "depression": 1,
    "irritability": 1,
    "visual_disturbances": 1
  },
  "Cervical spondylosis": {
    "back_pain": 1,
    "weakness_in_limbs": 1,
    "neck_pain": 1,
    "dizziness": 1,
    "loss_of_balance": 1
  },
  "Paralysis (brain hemorrhage)": {
    "vomiting": 1,
    "headache": 1,
    "weakness_of_one_body_side": 1,
    "altered_sensorium": 1
  },
  "Jaundice": {
    "itching": 1,
    "vomiting": 1,
    "fatigue": 1,
    "weight_loss": 1,
    "high_fever": 1,
    "yellowish_skin": 1,
    "dark_urine": 1,
    "abdominal_pain": 1
  },
  "Malaria": {
    "chills": 1,
    "vomiting": 1,
    "high_fever": 1,
    "sweating": 1,
    "headache": 1,
    "nausea": 1,
    "diarrhoea": 1,
    "muscle_pain": 1
  },
  "Chicken pox": {
    "itching": 1,
    "skin_rash": 1,
    "fatigue": 1,
    "lethargy": 1,
    "high_fever": 1,
    "headache": 1,
    "loss_of_appetite": 1,
    "mild_fever": 1,
    "swelled_lymph_nodes": 1,
    "malaise": 1,
    "red_spots_over_body": 1
  },
  "Dengue": {
    "skin_rash": 1,
    "chills": 1,
    "joint_pain": 1,
    "vomiting": 1,
    "fatigue": 1,
    "high_fever": 1,
    "headache": 1,
    "nausea": 1,
    "loss_of_appetite": 1,
    "pain_behind_the_eyes": 1,
    "back_pain": 1,
    "malaise": 1,
    "muscle_pain": 1,
    "red_spots_over_body": 1
  },
  "Typhoid": {
    "chills": 1,
    "vomiting": 1,
    "fatigue": 1,
    "high_fever": 1,
    "headache": 1,
    "nausea": 1,
    "constipation": 1,
    "abdominal_pain": 1,
    "diarrhoea": 1,
    "toxic_look_(typhos)": 1,
    "belly_pain": 1
  },
  "hepatitis A": {
    "joint_pain": 1,
    "vomiting": 1,
    "yellowish_skin": 1,
    "dark_urine": 1,
    "nausea": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "diarrhoea": 1,
    "mild_fever": 1,
    "yellowing_of_eyes": 1,
    "muscle_pain": 1
  },
  "Hepatitis B": {
    "itching": 1,
    "fatigue": 1,
    "lethargy": 1,
    "yellowish_skin": 1,
    "dark_urine": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "yellow_urine": 1,
    "yellowing_of_eyes": 1,
    "malaise": 1,
    "receiving_blood_transfusion": 1,
    "receiving_unsterile_injections": 1
  },
  "Hepatitis C": {
    "fatigue": 1,
    "yellowish_skin": 1,
    "nausea": 1,
    "loss_of_appetite": 1,
    "yellowing_of_eyes": 1,
    "family_history": 1
  },
  "Hepatitis D": {
    "joint_pain": 1,
    "vomiting": 1,
    "fatigue": 1,
    "yellowish_skin": 1,
    "dark_urine": 1,
    "nausea": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "yellowing_of_eyes": 1
  },
  "Hepatitis E": {
    "joint_pain": 1,
    "vomiting": 1,
    "fatigue": 1,
    "high_fever": 1,
    "yellowish_skin": 1,
    "dark_urine": 1,
    "nausea": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "yellowing_of_eyes": 1,
    "acute_liver_failure": 1,
    "coma": 1,
    "stomach_bleeding": 1
  },
  "Alcoholic hepatitis": {
    "vomiting": 1,
    "yellowish_skin": 1,
    "abdominal_pain": 1,
    "swelling_of_stomach": 1,
    "distention_of_abdomen": 1,
    "history_of_alcohol_consumption": 1,
    "fluid_overload.1": 1
  },
  "Tuberculosis": {
    "chills": 1,
    "vomiting": 1,
    "fatigue": 1,
    "weight_loss": 1,
    "cough": 1,
    "high_fever": 1,
    "breathlessness": 1,
    "sweating": 1,
    "loss_of_appetite": 1,
    "mild_fever": 1,
    "yellowing_of_eyes": 1,
    "swelled_lymph_nodes": 1,
    "malaise": 1,
    "phlegm": 1,
    "chest_pain": 1,
    "blood_in_sputum": 1
  },
  "Common Cold": {
    "continuous_sneezing": 1,
    "chills": 1,
    "fatigue": 1,
    "cough": 1,
    "high_fever": 1,
    "headache": 1,
    "swelled_lymph_nodes": 1,
    "malaise": 1,
    "phlegm": 1,
    "throat_irritation": 1,
    "redness_of_eyes": 1,
    "sinus_pressure": 1,
    "runny_nose": 1,
    "congestion": 1,
    "chest_pain": 1,
    "loss_of_smell": 1,
    "muscle_pain": 1
  },
  "Pneumonia": {
    "chills": 1,
    "fatigue": 1,
    "cough": 1,
    "high_fever": 1,
    "breathlessness": 1,
    "sweating": 1,
    "malaise": 1,
    "phlegm": 1,
    "chest_pain": 1,
    "fast_heart_rate": 1,
    "rusty_sputum": 1
  }
}


# Convert to DataFrame
symptom_columns = list(set(symptom for symptoms in data.values() for symptom in symptoms.keys()))
X = []
y = []

for disease, symptoms in data.items():
    symptom_values = [symptoms.get(symptom, 0) for symptom in symptom_columns]
    X.append(symptom_values)
    y.append(disease)

X = pd.DataFrame(X, columns=symptom_columns)
y = pd.Series(y)

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Save the model and encoder
import pickle

with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)
