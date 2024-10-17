from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Sample dataset of diseases and their symptoms
disease_data = {
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
chronic_diseases = {
  "Dimorphic hemmorhoids(piles)": {
    "constipation": 1,
    "pain_during_bowel_movements": 1,
    "pain_in_anal_region": 1,
    "bloody_stool": 1,
    "irritation_in_anus": 1
  },
  "Heart attack": {
    "vomiting": 1,
    "breathlessness": 1,
    "sweating": 1,
    "chest_pain": 1
  },
  "Varicose veins": {
    "fatigue": 1,
    "cramps": 1,
    "bruising": 1,
    "obesity": 1,
    "swollen_legs": 1,
    "swollen_blood_vessels": 1,
    "prominent_veins_on_calf": 1
  },
  "Hypothyroidism": {
    "fatigue": 1,
    "weight_gain": 1,
    "cold_hands_and_feets": 1,
    "mood_swings": 1,
    "lethargy": 1,
    "dizziness": 1,
    "puffy_face_and_eyes": 1,
    "enlarged_thyroid": 1,
    "brittle_nails": 1,
    "swollen_extremeties": 1,
    "depression": 1,
    "irritability": 1,
    "abnormal_menstruation": 1
  },
  "Hyperthyroidism": {
    "fatigue": 1,
    "mood_swings": 1,
    "weight_loss": 1,
    "restlessness": 1,
    "sweating": 1,
    "diarrhoea": 1,
    "fast_heart_rate": 1,
    "excessive_hunger": 1,
    "muscle_weakness": 1,
    "irritability": 1,
    "abnormal_menstruation": 1
  },
  "Hypoglycemia": {
    "vomiting": 1,
    "fatigue": 1,
    "anxiety": 1,
    "sweating": 1,
    "headache": 1,
    "nausea": 1,
    "blurred_and_distorted_vision": 1,
    "excessive_hunger": 1,
    "drying_and_tingling_lips": 1,
    "slurred_speech": 1,
    "irritability": 1,
    "palpitations": 1
  },
  "Osteoarthristis": {
    "joint_pain": 1,
    "neck_pain": 1,
    "knee_pain": 1,
    "hip_joint_pain": 1,
    "swelling_joints": 1,
    "painful_walking": 1
  },
  "Arthritis": {
    "muscle_weakness": 1,
    "stiff_neck": 1,
    "swelling_joints": 1,
    "movement_stiffness": 1,
    "painful_walking": 1
  }
}

def predict_disease(disease_data, payload):
    matched_diseases = {}

    # Iterate over each disease in the dataset
    for disease, symptoms in disease_data.items():
        match_count = 0

        # Count how many symptoms match the payload
        for symptom in symptoms:
            if symptom in payload and payload[symptom] == 1:
                match_count += 1

        # If there are matching symptoms, add to the results
        if match_count > 0:
            matched_diseases[disease] = match_count

    # Sort diseases by the number of matching symptoms (descending)
    sorted_diseases = sorted(matched_diseases.items(), key=lambda x: x[1], reverse=True)

    # Return the predicted disease or "No disease"
    if sorted_diseases:
        return {"predicted_disease": sorted_diseases[0][0]}
    else:
        return {"predicted_disease": "no_infection_disease"}

@app.route('/predict-infection', methods=['POST'])
def predict_infection():

    data = request.json
    result = predict_disease(disease_data, data)
    return jsonify(result)





def predict_conditions_disease(disease_data, payload):
    matched_diseases = {}

    # Iterate over each disease in the dataset
    for disease, symptoms in disease_data.items():
        match_count = 0

        # Count how many symptoms match the payload
        for symptom in symptoms:
            if symptom in payload and payload[symptom] == 1:
                match_count += 1

        # If there are matching symptoms, add to the results
        if match_count > 0:
            matched_diseases[disease] = match_count

    # Sort diseases by the number of matching symptoms (descending)
    sorted_diseases = sorted(matched_diseases.items(), key=lambda x: x[1], reverse=True)

    # Return the predicted disease or "No disease"
    if sorted_diseases:
        return {"predicted_disease": sorted_diseases[0][0]}
    else:
        return {"predicted_disease": "no_chronic_diseases"}

@app.route('/predict-conditions', methods=['POST'])
def predict_conditions():
    data = request.json
    result = predict_conditions_disease(chronic_diseases, data)
    return jsonify(result)







@app.route('/predict-heart', methods=['POST'])
def predict_heart():
    try:
        data = request.get_json()

        # Extract the values from the JSON payload
        age = int(data['age'])
        sex = 1 if data['gender'].lower() == 'female' else 0  # Female: 1, Male: 0
        cp = int(data['cp'])
        trestbps = int(data['Trestbps'])
        chol = int(data['cholesterol'])


        # Perform prediction using machine learning model
        prediction = model.predict([[age, sex, cp, trestbps, chol]])
        if prediction[0] == 0:
            return jsonify({"predicted_disease": "no_heart_disease"})
        else:
            return jsonify({"predicted_disease": "Heart Disease"})
           
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
