import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# --- Download necessary NLTK data ---
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Preprocessing Functions ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
negation_words = ["not", "no", "never", "n't", "cannot"]

def handle_negation(tokens):
    modified_tokens = []
    negate = False
    for i, token in enumerate(tokens):
        if token in negation_words:
            negate = True
        elif negate:
            if token in ["only", "just", "very", "much"]:
                modified_tokens.append(token)
            else:
                modified_tokens.append(token + "_NEG")
                negate = False
        else:
            modified_tokens.append(token)
    return modified_tokens

def preprocess_text(text, steps):
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\\w+|\\#', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    tokens = word_tokenize(text)

    if 'remove_stopwords' in steps:
        tokens = [word for word in tokens if word not in stop_words]
    if 'stemming' in steps:
        tokens = [stemmer.stem(word) for word in tokens]
    if 'lemmatization' in steps:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    if 'handle_negation' in steps:
        tokens = handle_negation(tokens)
    return " ".join(tokens)

# --- Model and Preprocessing Object Loading Functions ---

def load_all_models(models_dir):
    """Loads all models from the specified directory."""
    all_models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith(".h5"):
            model_path = os.path.join(models_dir, filename)
            model_name = os.path.splitext(filename)[0]
            try:
                model = load_model(model_path)
                model_info = {
                    'model': model,
                    'type': 'dl',
                    'name': model_name.split('_')[0]
                }
                all_models[model_name] = model_info
            except Exception as e:
                print(f"Error loading DL model {filename}: {e}")

        elif filename.endswith(".joblib") and "vectorizer" not in filename and "tokenizer" not in filename and "label_binarizer" not in filename:
            model_path = os.path.join(models_dir, filename)
            model_name_full = os.path.splitext(filename)[0]
            try:
                model = joblib.load(model_path)
                name_parts = model_name_full.split('_')
                
                feature_method = name_parts[-1]
                model_type_name = name_parts[0]
                # The preprocessing steps form the middle part
                preprocessing_steps_str = '_'.join(name_parts[1:-1])

                model_info = {
                    'model': model,
                    'type': 'ml',
                    'name': model_type_name,
                    'feature_method': feature_method,
                    'preprocessing_steps_str': preprocessing_steps_str
                }
                all_models[model_name_full] = model_info
            except Exception as e:
                print(f"Error loading ML model {filename}: {e}")
    return all_models

def load_vectorizers_and_binarizers(models_dir):
    """Loads all vectorizers, tokenizers, and label binarizers."""
    vectorizers = {}
    tokenizers = {}
    label_binarizers = {}

    for filename in os.listdir(models_dir):
        if filename.endswith(".joblib"):
            path = os.path.join(models_dir, filename)
            name = os.path.splitext(filename)[0]
            if "vectorizer" in name:
                vectorizers[name.replace('_vectorizer', '')] = joblib.load(path)
            elif "tokenizer" in name:
                tokenizers[name.replace('_tokenizer', '')] = joblib.load(path)
            elif "label_binarizer" in name:
                label_binarizers[name.replace('_label_binarizer', '')] = joblib.load(path)

    return vectorizers, tokenizers, label_binarizers 