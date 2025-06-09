# test_auc_text.py
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics      import roc_auc_score, accuracy_score
from keras.models         import load_model

# --- Load dữ liệu đã tiền-xử lý và nhãn ---
with open('data/text_processed.pkl', 'rb') as f:
    text_data = pickle.load(f)       # dict: {ID: vector}
orig = pd.read_csv('data/ids_raw_texts_labels.csv', index_col='ID')
valid_ids = list(text_data.keys())
X = np.array([text_data[i] for i in valid_ids])
y = np.array([orig.loc[i, 'Labels'] for i in valid_ids])

# --- Chia train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Load model và predict ---
model = load_model('checkpoints/text_model_final.h5', compile=False)
preds = model.predict(X_test).ravel()

# --- Tính AUC ---
auc = roc_auc_score(y_test, preds)
print(f'Text-only AUC:      {auc:.4f}')

# --- Tính Accuracy ---
# Dùng threshold = 0.5 để xác định nhãn dự đoán
preds_label = (preds >= 0.5).astype(int)
acc = accuracy_score(y_test, preds_label)
print(f'Text-only Accuracy: {acc:.4f}')
