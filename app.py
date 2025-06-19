from flask import Flask, render_template, request, jsonify
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# It's better to move preprocessing functions to a separate file
from utils import preprocess_text, load_all_models, load_vectorizers_and_binarizers

app = Flask(__name__)

# --- Load Models and Preprocessing Objects ---
# Get the absolute path to the saved_models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
all_models = load_all_models(models_dir)
vectorizers, tokenizers, label_binarizers = load_vectorizers_and_binarizers(models_dir)


@app.route('/')
def index():
    # Get a list of model names to display in the dropdown
    model_names = list(all_models.keys())
    return render_template('index.html', models=model_names)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        selected_models = request.form.getlist('models')
        
        predictions = {}

        for model_name in selected_models:
            model_info = all_models.get(model_name)
            if not model_info:
                predictions[model_name] = "Model not found."
                continue
            
            model = model_info['model']
            model_type = model_info['type']
            
            # --- Preprocessing based on model name ---
            preprocessing_steps_str = ""
            if model_type == 'dl':
                # For DL model 'CNN_[]', split gives ['CNN', '[]'], so [1] is '[]'
                try:
                    preprocessing_steps_str = model_name.split('_', 1)[1]
                except IndexError:
                    preprocessing_steps_str = "[]"
            elif model_type == 'ml':
                preprocessing_steps_str = model_info.get('preprocessing_steps_str', "[]")
            
            try:
                preprocessing_steps = json.loads(preprocessing_steps_str.replace("'", "\""))
            except json.JSONDecodeError:
                preprocessing_steps = []

            processed_text = preprocess_text(text, preprocessing_steps)
            
            # --- Prediction ---
            if model_type == 'dl':
                # For deep learning models
                tokenizer_key = f"{model_info['name']}_{preprocessing_steps_str}"
                tokenizer = tokenizers.get(tokenizer_key)
                label_binarizer = label_binarizers.get(tokenizer_key)
                
                if tokenizer and label_binarizer:
                    sequence = tokenizer.texts_to_sequences([processed_text])
                    padded_sequence = pad_sequences(sequence, maxlen=100)
                    prediction = model.predict(padded_sequence)
                    predicted_label = label_binarizer.inverse_transform(prediction)[0]
                    predictions[model_name] = predicted_label
                else:
                    predictions[model_name] = f"Tokenizer/Binarizer not found. Looked for: {tokenizer_key}"

            elif model_type == 'ml':
                 # For machine learning models
                vectorizer_key = f"{model_info['name']}_{preprocessing_steps_str}_{model_info['feature_method']}"
                vectorizer = vectorizers.get(vectorizer_key)

                if vectorizer:
                    feature_vector = vectorizer.transform([processed_text])
                    prediction = model.predict(feature_vector)
                    predictions[model_name] = prediction[0]
                else:
                    predictions[model_name] = f"Vectorizer not found. Looked for: {vectorizer_key}"

        return render_template('index.html', predictions=predictions, text=text, models=list(all_models.keys()), selected_models=selected_models)

if __name__ == '__main__':
    app.run(debug=True) 