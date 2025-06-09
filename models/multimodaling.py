# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pickle
import cnn_model
import json
import numpy as np
import pandas as pd
from keras.optimizers import Adadelta, Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.applications.densenet import DenseNet121
from keras import backend as K
from keras.layers import Merge
from keras.layers import Input, Dense, Conv1D, Dropout, Flatten, Conv2D, MaxPooling1D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Embedding
from keras.models import Model

# %% [markdown]
# ## Modeling : Multimodel (Text +Image Fusion)
#
# This section focuses on the multimodal
# architecture which consists of a text submodel and
# an image submodel with an emphisis on transfer learning. The text submodel is developed by taking the network layers, from input layers to feature vectors, from a trained text classifier. Similarly, the image sub-model takes the network layers of a trained image classifier from input layers to feature vectors.
#
# Then the encoded text and image feature vectors are concatenated into a single flat feature vector.
# This feature vector is then passed onto a densely connected decoder for the binary classification. Applying transfer learning to the two encoders makes this multimodal function under small data situations such as this one (below 3K). The transferred text and image encoders are the respective pre-trained embedding and residual layers. 
#
# pretrained encoders are then finely tuned on low learning rates.

# %% [markdown]
# ### Load data và chia làm train/test

# %%
with open('data/text_processed.pkl', 'rb') as handle:
    text = pickle.load(handle)
    
with open('data/x_ray_processed.pkl', 'rb') as handle:
    img = pickle.load(handle)
    
with open('data/vocab.json', 'r') as voc:
    vocab = json.load(voc) 
    
original_data = pd.read_csv('data/ids_raw_texts_labels.csv')
# taking the intersection of ids of both datasets
ids   = list(set(list(text.keys())) & set(list(img.keys())))
text  = [text[patient] for patient in ids]
img   = [img[patient] for patient in ids]
y     = [original_data[original_data['ID'] == patient].Labels.item() for patient in ids]

# Split the dataset for text
X_train_text, X_test_text, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42)

# Split the dataset for img
X_train_img, X_test_img, y_train, y_test = train_test_split(img, y, test_size=0.2, random_state=42)

# random state is the same so same id splits go to both types of datasets

# %% [markdown]
# ## Text model train :Text CNN
#
# We choose a CNN based text classifier (Kim,2015) to meet this task. We also extend the 1-D CNN for classifying short e-commerce product descriptions (Eskesen, 2017) to our text classifier. Resemblance in structure of product descriptions, features of short radiology reports (27.4 words per case on average) are extracted by the Word2Vec approach (Mikolov, 2013).
#
# The transfered embedding layer is a pretrained word2vec model fine-tuned on a large medical corpus. The word embeddings can be found under 'data/new_data_embed300.txt'.

# %%
# EMBEDDING
MAX_NUM_WORDS  = 15000
EMBEDDING_DIM  = 300
MAX_SEQ_LENGTH = 140
USE_GLOVE      = True

# MODEL
FILTER_SIZES   = [3,4,5]
FEATURE_MAPS   = [10,10,10]
DROPOUT_RATE   = 0.5

# LEARNING
BATCH_SIZE     = 200
NB_EPOCHS      = 10
RUNS           = 1


def create_glove_embeddings():
    embeddings_index = {}
    f = open('data/new_data_embed300.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    for word, i in vocab.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM,
                     input_length=MAX_SEQ_LENGTH,
                     weights=[embedding_matrix],
                     trainable=True
                    )


# %%
histories = []

for i in range(RUNS):
    print('Running iteration %i/%i' % (i+1, RUNS))
        
    emb_layer = None
    if USE_GLOVE:
        emb_layer = create_glove_embeddings()
    
    model = cnn_model.build_cnn(
        embedding_layer=emb_layer,
        num_words=MAX_NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        filter_sizes=FILTER_SIZES,
        feature_maps=FEATURE_MAPS,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout_rate=DROPOUT_RATE
    )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adadelta(clipvalue=3),
        metrics=['accuracy']
    )
    
    history = model.fit(
        np.array(X_train_text), y_train,
        epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(np.array(X_test_text), y_test),
        callbacks=[ModelCheckpoint('best_models/text_model-%i.h5'%(i+1), monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='min'),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01)
                  ]
    )
    print()
    histories.append(history.history)

# %% [markdown]
# ## Image Model train:  DenseNet121
#
# The image submodal is centered around the idea of transfer learning. CNN encoders pre-trained on ImageNet and National Health Institutes ChestX-ray14 gave several different experimentation combinations with diferent encoders (VGG, DenseNet, ResNet, etc.). The DenseNet121 (He et al., 2015) pre-trained on ChestX-ray14 was chosen as the encoder due to its superior accuracy. This encoder is then fed into a simple batch normalization which is then fed into a decoder for binary classification.

# %%
image_input = Input(shape=(224, 224, 3))
base_model = DenseNet121(include_top=True, weights='best_models/CheXNet_Densenet121_weights.h5', input_tensor=image_input, input_shape=None, pooling=None, classes=14)
last_layer = base_model.get_layer('avg_pool').output
x = BatchNormalization()(last_layer)
x = Dense(512, activation='relu')(x)
x = Dropout(.5)(x)
x = BatchNormalization()(x)
out = Dense(1, activation='softmax')(x) 
model = Model(image_input , out)

# %%
model.compile(loss='binary_crossentropy',optimizer=Adam(), metrics=['accuracy'])    
hist = model.fit(np.array(X_train_img), np.array(y_train), batch_size=32, epochs=3, verbose=1, validation_data=(np.array(X_test_img), np.array(y_test)))
#model.save_weights('best_models/img_model-1.h5')

# %% [markdown]
# ## Multimodal train
#

# %%
###########################################
# Text Sub-model
def create_channel(x, filter_size, feature_map):
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    x = Flatten()(x)
    return x
x_in = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
channels = []
embedding_layer = create_glove_embeddings()
emb_layer = embedding_layer(x_in)
if DROPOUT_RATE:
    emb_layer = Dropout(DROPOUT_RATE)(emb_layer)
for ix in range(len(FILTER_SIZES)):
    x = create_channel(emb_layer, FILTER_SIZES[ix], FEATURE_MAPS[ix])
    channels.append(x)
# Concatenate all channels
x = concatenate(channels)
text_last_layer = concatenate(channels)
###########################################
# Image Sub-model
image_input = Input(shape=(224, 224, 3))
base_model = DenseNet121(include_top=True, weights='best_models/CheXNet_Densenet121_weights.h5', input_tensor=image_input, input_shape=None, pooling=None, classes=14)
last_layer = base_model.get_layer('avg_pool').output
img_last_layer = BatchNormalization()(last_layer)
###########################################
# Fusion
fusion = concatenate([text_last_layer,img_last_layer])
x = BatchNormalization()(fusion)
x = Dense(512, activation='relu')(x)
x = Dropout(.3)(x)
x = BatchNormalization()(x)
out = Dense(1, activation='softmax')(x)
multi_model = Model([x_in, image_input] , out)
###########################################

# %%
# multi_model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])    
hist = multi_model.fit(x = ([np.array(X_train_text),np.array(X_train_img)]), y = np.array(y_train), batch_size=32, epochs=3, verbose=1, validation_data=(([np.array(X_test_text),np.array(X_test_img)]), np.array(y_test)))
#model.save_weights('best_models/multi_model-1.h5')
