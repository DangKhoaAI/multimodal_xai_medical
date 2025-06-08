import os
import re
import numpy as np
import pandas as pd
import nltk
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     Flatten, Embedding)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

# ------------------------
# Configuration Parameters
# ------------------------
# Paths
IMAGE_DIR       = '../NLMCXR_png/'
LABEL_CSV       = '../ids_texts_labels.csv'
GLOVE_PATH      = '/path/to/glove.300d.txt'  # update as needed

# Text settings
MAX_NUM_WORDS   = 15000
EMBEDDING_DIM   = 300
MAX_SEQ_LENGTH  = 140
FILTER_SIZES    = [3, 4, 5]
FEATURE_MAPS    = [10, 10, 10]
DROPOUT_RATE    = 0.5
BATCH_SIZE      = 200
NB_EPOCHS       = 10
VALIDATION_SPLIT= 0.2
USE_GLOVE       = True

# Image settings
IMG_TARGET_SIZE = (224, 224)
BATCH_SIZE_IMG  = 32
IMG_EPOCHS      = 10

# ------------------------
# Text Processing Utilities
# ------------------------
def clean_doc(doc: str) -> str:
    """
    Clean a single document: lowercase, remove numbers, stopwords, punctuation, short words.
    """
    doc = doc.lower()
    doc = re.sub(r"\d+", "", doc)
    tokens = text_to_word_sequence(doc)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if len(w) > 1]
    return ' '.join(tokens)


def load_text_data(csv_path: str, ids: list) -> tuple:
    df = pd.read_csv(csv_path)
    texts, labels = [], []
    for idx in ids:
        row = df[df['ID'] == idx]
        if not row.empty:
            texts.append(str(row['Text'].iloc[0]))
            labels.append(int(row['Labels'].iloc[0]))
    return texts, labels


def prepare_text_sequences(train_texts, test_texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_texts)
    train_seq = tokenizer.texts_to_sequences(train_texts)
    test_seq  = tokenizer.texts_to_sequences(test_texts)
    X_train = pad_sequences(train_seq, maxlen=MAX_SEQ_LENGTH, padding='post')
    X_test  = pad_sequences(test_seq,  maxlen=MAX_SEQ_LENGTH, padding='post')
    return X_train, X_test, tokenizer.word_index


def load_glove_embeddings(word_index):
    embeddings_index = {}
    with open(GLOVE_PATH, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < MAX_NUM_WORDS:
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec

    return Embedding(input_dim=MAX_NUM_WORDS,
                     output_dim=EMBEDDING_DIM,
                     input_length=MAX_SEQ_LENGTH,
                     weights=[embedding_matrix],
                     trainable=True)

# ------------------------
# Image Processing Utilities
# ------------------------
def load_image_ids(image_dir: str) -> list:
    imgs = os.listdir(image_dir)
    return [f for f in imgs if f.lower().endswith('.png')]


def load_image_data(image_dir: str, filenames: list) -> np.ndarray:
    data = []
    for fname in filenames:
        img_path = os.path.join(image_dir, fname)
        img = keras_image.load_img(img_path, target_size=IMG_TARGET_SIZE)
        x   = keras_image.img_to_array(img)
        x   = preprocess_input(x)
        data.append(x)
    return np.array(data)

# ------------------------
# Model Builders
# ------------------------
def build_text_cnn(embedding_layer=None):
    from cnn_model import build_cnn

    return build_cnn(
        embedding_layer=embedding_layer,
        num_words=MAX_NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        filter_sizes=FILTER_SIZES,
        feature_maps=FEATURE_MAPS,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout_rate=DROPOUT_RATE
    )


def build_image_model():
    input_img = Input(shape=(*IMG_TARGET_SIZE, 3))
    base = DenseNet121(
        include_top=True,
        weights='imagenet',
        input_tensor=input_img,
        classes=14
    )
    x = base.get_layer('avg_pool').output
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=output)

    # Freeze base layers
    for layer in model.layers[:-4]:
        layer.trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    return model

# ------------------------
# Main Execution
# ------------------------
def main():
    # Prepare image data
    image_files = load_image_ids(IMAGE_DIR)
    ids_numeric = [re.findall(r'CXR(.*)_', f)[0] for f in image_files]
    images = load_image_data(IMAGE_DIR, image_files)

    df = pd.read_csv(LABEL_CSV)
    labels_img = [int(df[df['ID'] == f' CXR{idx} '].Labels) for idx in ids_numeric]

    X_img, X_img_test, y_img, y_img_test = train_test_split(
        images, labels_img, test_size=VALIDATION_SPLIT, random_state=31
    )

    # Build and train image model
    img_model = build_image_model()
    img_model.fit(
        X_img, y_img,
        epochs=IMG_EPOCHS,
        batch_size=BATCH_SIZE_IMG,
        validation_data=(X_img_test, y_img_test)
    )

    # Prepare text data
    texts, text_labels = load_text_data(LABEL_CSV, ids_numeric)
    clean_texts = [clean_doc(t) for t in texts]

    # Split text data
    X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
        clean_texts, text_labels, test_size=VALIDATION_SPLIT, random_state=31
    )
    X_train_seq, X_test_seq, word_index = prepare_text_sequences(X_train_txt, X_test_txt)

    # Embedding layer
    emb_layer = None
    if USE_GLOVE:
        emb_layer = load_glove_embeddings(word_index)

    # Build and train text model
    txt_model = build_text_cnn(embedding_layer=emb_layer)
    txt_model.compile(
        loss='binary_crossentropy',
        optimizer=Adadelta(clipvalue=3),
        metrics=['accuracy']
    )
    txt_model.fit(
        X_train_seq, y_train_txt,
        epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_seq, y_test_txt),
        callbacks=[
            ModelCheckpoint('text_model.h5', monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01)
        ]
    )


if __name__ == '__main__':
    main()
