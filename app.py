from keras.models import load_model
from config import IMG_MODEL_PATH, TXT_MODEL_PATH, MRG_MODEL_PATH

import io
import os
import base64

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input

from config import MAX_NUM_WORDS, MAX_SEQ_LENGTH, DATA_DIR, IMG_TARGET_SIZE
from models.load_models import load_all_models
from explainability.image_explainer import explain_image_with_gradients
from explainability.text_explainer import explain_text_with_gradients

app = Flask(__name__)

# --- 1. Load models once ---
models = load_all_models()
text_model = models['text']
image_model = models['image']
multimodal_model = models['multimodal']

# --- 2. Build tokenizer on raw texts ---
raw_df = pd.read_csv(os.path.join(DATA_DIR, 'ids_raw_texts_labels.csv'))
all_texts = raw_df['Text'].astype(str).tolist()

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(all_texts)

# --- 3. Tab configuration ---
TAB_CONFIG = {
    'ai':        {'name': 'AI in Healthcare',      'template': 'tabs/ai_healthcare.html'},
    'xai':       {'name': 'XAI',                    'template': 'tabs/xai.html'},
    'inference': {'name': 'Model Inference',        'template': 'tabs/inference.html'},
}


def encode_image(img: Image.Image) -> str:
    """Convert PIL image to base64-encoded PNG string"""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def predict_text_xai(text: str) -> tuple:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post')
    prob = text_model.predict(padded)[0][0]
    fig_html = explain_text_with_gradients(text_model, padded, text)
    return f"{prob:.4f}", fig_html


def predict_image_xai(file_stream) -> tuple:
    pil = Image.open(file_stream).convert('RGB').resize(IMG_TARGET_SIZE)
    arr = np.array(pil)
    x = preprocess_input(np.expand_dims(arr, 0))
    prob = image_model.predict(x)[0][0]
    xai_img, _ = explain_image_with_gradients(x[0], image_model)
    orig_b64 = encode_image(pil)
    xai_b64 = encode_image(Image.fromarray(np.clip(xai_img, 0, 255).astype('uint8')))
    return f"{prob:.4f}", orig_b64, xai_b64


def predict_multimodal_xai(text: str, file_stream) -> tuple:
    # Prepare text input
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post')
    # Prepare image input
    pil = Image.open(file_stream).convert('RGB').resize(IMG_TARGET_SIZE)
    arr = np.array(pil)
    x = preprocess_input(np.expand_dims(arr, 0))
    # Multimodal prediction
    prob = multimodal_model.predict([x, padded])[0][0]
    # Explanations
    text_fig = explain_text_with_gradients(text_model, padded, text)
    xai_img, _ = explain_image_with_gradients(x[0], image_model)
    orig_b64 = encode_image(pil)
    xai_b64 = encode_image(Image.fromarray(np.clip(xai_img, 0, 255).astype('uint8')))
    return f"{prob:.4f}", text_fig, orig_b64, xai_b64


@app.route('/', defaults={'active_tab_id': 'ai'}, methods=['GET', 'POST'])
@app.route('/<active_tab_id>', methods=['GET', 'POST'])
def index(active_tab_id):
    if active_tab_id not in TAB_CONFIG:
        active_tab_id = 'ai'

    form_data = {'mode': 'text', 'text_data': ''}
    results = {}

    if request.method == 'POST' and active_tab_id == 'inference':
        mode = request.form.get('mode')
        form_data['mode'] = mode

        if mode == 'text':
            text = request.form.get('text_data', '').strip()
            form_data['text_data'] = text
            results['text_pred'], results['text_explanation'] = predict_text_xai(text)

        elif mode == 'image':
            file = request.files.get('image_data')
            if file:
                img_pred, orig_b64, xai_b64 = predict_image_xai(file.stream)
                results['image_pred'] = img_pred
                results['orig_image'] = orig_b64
                results['xai_image'] = xai_b64

        elif mode == 'both':
            text = request.form.get('text_data', '').strip()
            form_data['text_data'] = text
            file = request.files.get('image_data')
            if file:
                both_pred, both_text_fig, orig_b64, xai_b64 = predict_multimodal_xai(text, file.stream)
                results['both_pred'] = both_pred
                results['text_explanation'] = both_text_fig
                results['orig_image'] = orig_b64
                results['xai_image'] = xai_b64

    return render_template(
        'index.html',
        tabs=TAB_CONFIG,
        active_tab_id=active_tab_id,
        form_data=form_data,
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True)
