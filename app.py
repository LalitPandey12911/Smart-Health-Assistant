from flask import Flask, send_file, render_template, request, redirect, url_for
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import pickle
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
from sklearn.metrics import accuracy_score

# Setup Flask and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Directories
MODEL_DIR = 'models'
DATA_DIR = 'data'

# Globals
last_prediction_data = {}
last_disease_probs = {}

# === Load Data & Models ===
def safe_load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)

def load_models():
    return {
        "svc": pickle.load(open(os.path.join(MODEL_DIR, "svc.pkl"), "rb")),
        "rf": pickle.load(open(os.path.join(MODEL_DIR, "rf.pkl"), "rb")),
        "label_encoder": pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    }

def load_data():
    data = {
        "description": safe_load_csv('description.csv'),
        "precautions": safe_load_csv('precautions_df.csv'),
        "medications": safe_load_csv('medications.csv'),
        "diets": safe_load_csv('diets.csv'),
        "workout": safe_load_csv('workout_df.csv'),
        "symptoms": safe_load_csv('symtoms_df.csv'),
        "training": safe_load_csv("Training.csv")
    }

    # Strip whitespace from 'Disease' column in all DataFrames
    for key in data:
        if 'Disease' in data[key].columns:
            data[key]['Disease'] = data[key]['Disease'].str.strip()

    return data

models = load_models()
data = load_data()
training_data = data["training"]
all_symptoms = training_data.columns[:-1].tolist()

# === Core Logic ===
def correct_symptom(symptom):
    return str(TextBlob(symptom).correct()).lower()

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(all_symptoms))
    unrecognized = []

    for s in symptoms_list:
        corrected = correct_symptom(s)
        if corrected in all_symptoms:
            index = all_symptoms.index(corrected)
            input_vector[index] = 1
        else:
            unrecognized.append(s)

    if unrecognized:
        logging.warning(f"Unrecognized symptoms: {', '.join(unrecognized)}")

    df_input = pd.DataFrame([input_vector], columns=all_symptoms)

    svc_pred = models['svc'].predict(df_input)[0]
    rf_pred = models['rf'].predict(df_input)[0]

    label_encoder = models['label_encoder']
    svc_pred_label = label_encoder.inverse_transform([svc_pred])[0]
    rf_pred_label = label_encoder.inverse_transform([rf_pred])[0]

    # Convert prognosis column if it's not already encoded
    y_true = label_encoder.transform(training_data['prognosis'])

    svc_acc = accuracy_score(y_true, models['svc'].predict(training_data[all_symptoms]))
    rf_acc = accuracy_score(y_true, models['rf'].predict(training_data[all_symptoms]))

    selected_model = 'rf' if rf_acc > svc_acc else 'svc'
    selected_pred = rf_pred if selected_model == 'rf' else svc_pred
    predicted_disease = label_encoder.inverse_transform([selected_pred])[0]
    model_used = "Random Forest" if selected_model == 'rf' else "SVC"

    rf_probs = models['rf'].predict_proba(df_input)[0]
    top_probs = dict(sorted(zip(models['rf'].classes_, rf_probs), key=lambda x: x[1], reverse=True)[:4])
    decoded_probs = {
        label_encoder.inverse_transform([int(cls)])[0]: round(prob, 4)
        for cls, prob in top_probs.items()
    }

    return predicted_disease, model_used, decoded_probs

def get_disease_info(disease):
    disease = disease.strip()  # Clean incoming disease string

    desc = data['description'].set_index('Disease').loc[disease, 'Description']
    pre = data['precautions'][data['precautions']['Disease'] == disease].iloc[0, 1:].dropna().tolist()
    med = data['medications'][data['medications']['Disease'] == disease]['Medication'].tolist()
    diet = data['diets'][data['diets']['Disease'] == disease]['Diet'].tolist()
    workout = data['workout'][data['workout']['disease'] == disease]['workout'].tolist()
    return desc, pre, med, diet, workout

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction_data, last_disease_probs
    name = request.form['name']
    age = request.form['age']
    location = request.form['location']
    symptoms_raw = request.form['symptoms']
    symptoms = [s.strip() for s in symptoms_raw.split(',') if s.strip()]

    if len(symptoms) < 3:
        return render_template('index.html', message="Please enter at least 3 symptoms.")

    result = predict_disease(symptoms)
    if isinstance(result, str):
        return render_template('index.html', message=result)

    predicted_disease, model_used, disease_probs = result
    dis_des, pre, med, diet, workout = get_disease_info(predicted_disease)

    last_prediction_data = {
        "Name": name,
        "Age": age,
        "Location": location,
        "Symptoms": ", ".join(symptoms),
        "Disease": predicted_disease,
        "Description": dis_des,
        "Precautions": ", ".join(pre),
        "Medications": ", ".join(med),
        "Diet Recommendations": ", ".join(diet),
        "Workouts": ", ".join(workout),
        "Model Used": model_used
    }
    last_disease_probs = disease_probs

    return redirect(url_for('result'))

@app.route('/result')
def result():
    if not last_prediction_data:
        return "No prediction data available.", 400
    logging.info(f"Prediction data: {last_prediction_data}")
    return render_template('results.html',
                           predicted_disease=last_prediction_data["Disease"],
                           dis_des=last_prediction_data["Description"],
                           my_precautions=last_prediction_data["Precautions"].split(', '),
                           medications=last_prediction_data["Medications"].split(', '),
                           my_diet=last_prediction_data["Diet Recommendations"].split(', '),
                           workout=last_prediction_data["Workouts"].split(', '),
                           model_used=last_prediction_data["Model Used"],
                           disease_probs=last_disease_probs)

@app.route('/probability_chart')
def probability_chart():
    if not last_disease_probs:
        return "No probability data to display.", 400
    logging.info(f"Disease probabilities: {last_disease_probs}")
    return render_template('probability_chart.html', disease_probs=last_disease_probs)

@app.route('/download_pdf')
def download_pdf():
    if not last_prediction_data:
        return "No prediction data available to generate report.", 400
    pdf_file = generate_prediction_pdf(last_prediction_data)
    return send_file(pdf_file, as_attachment=True, download_name="prediction_report.pdf", mimetype='application/pdf')

def generate_prediction_pdf(prediction_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Smart Health Assistant - Prediction Report")
    c.setFont("Helvetica", 12)
    y = height - 80

    for key, value in prediction_data.items():
        lines = value.split(', ') if isinstance(value, str) and ', ' in value else [value]
        c.drawString(50, y, f"{key}:")
        y -= 20
        for line in lines:
            c.drawString(70, y, f"- {line}")
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"New message from {name} ({email}): {message}")
        return render_template('contact.html', success=True)
    return render_template('contact.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy-policy.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  
    app.run(debug=False, host='0.0.0.0', port=port)
