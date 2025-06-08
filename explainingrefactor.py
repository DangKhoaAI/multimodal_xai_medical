# -*- coding: utf-8 -*-
"""
Refactored script for evaluating and explaining multimodal (text and image)
medical diagnosis models.

This script performs two main tasks:
1.  Evaluates the performance (AUC) of text-only, image-only, and
    multimodal models on a test set.
2.  Generates feature-level explanations for both image (X-ray) and text
    (doctor's notes) data to identify anomalous regions or words.
"""

import pickle
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import load_model
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as hcluster
import math
import plotly.figure_factory as ff
import plotly.io as pio

# --- Constants ---
# File paths
DATA_DIR = 'data/'
MODEL_DIR = 'best_models/'
TEXT_PROCESSED_PATH = f'{DATA_DIR}text_processed.pkl'
IMG_PROCESSED_PATH = f'{DATA_DIR}x_ray_processed.pkl'
ORIGINAL_DATA_PATH = f'{DATA_DIR}ids_raw_texts_labels.csv'

IMG_MODEL_PATH = f'{MODEL_DIR}img_model_final.h5'
TXT_MODEL_PATH = f'{MODEL_DIR}text_model_final.h5'
MRG_MODEL_PATH = f'{MODEL_DIR}full_model_final.h5'

IMG_WEIGHTS_PATH = f'{MODEL_DIR}image_weights_final.hdf5'
TXT_WEIGHTS_PATH = f'{MODEL_DIR}textfinal_weights_best.hdf5'
MRG_WEIGHTS_PATH = f'{MODEL_DIR}full_weights_best_final.hdf5'

# Model & Data Parameters
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# Explainability Parameters
ABNORMAL_PATIENT_IDX = 125
NORMAL_PATIENT_IDX = 120
IG_GRADIENT_THRESHOLD = 0.70
LESION_CLUSTER_THRESHOLD = 10
LESION_CIRCLE_RADIUS_PADDING = 5

# Set Plotly to render in browser
# Note: For offline use or in different environments, you might use other renderers.
pio.renderers.default = "browser"


# Note: The original code depended on a custom `IntegratedGradients.py` module.
# We are assuming it's available in the execution path.
from IntegratedGradients import integrated_gradients


def load_and_prepare_data():
    """
    Loads processed text, image, and raw text data, finds the intersection
    of patient IDs, and splits the data into training and testing sets.

    Returns:
        A dictionary containing all data splits.
    """
    print("Loading and preparing data...")
    with open(TEXT_PROCESSED_PATH, 'rb') as handle:
        text_data = pickle.load(handle)
    with open(IMG_PROCESSED_PATH, 'rb') as handle:
        img_data = pickle.load(handle)

    original_data = pd.read_csv(ORIGINAL_DATA_PATH)
    original_data.set_index('ID', inplace=True)

    # Find intersection of patient IDs
    valid_ids = list(set(text_data.keys()) & set(img_data.keys()))

    # Align data based on valid IDs
    raw_text = [original_data.loc[pid, 'Text'] for pid in valid_ids]
    processed_text = [text_data[pid] for pid in valid_ids]
    processed_img = [img_data[pid] for pid in valid_ids]
    labels = [original_data.loc[pid, 'Labels'] for pid in valid_ids]

    # Split datasets with the same random state for consistency
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        processed_text, labels, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )
    X_train_img, X_test_img, _, _ = train_test_split(
        processed_img, labels, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )
    _, X_test_raw_text, _, _ = train_test_split(
        raw_text, labels, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    return {
        "X_train_text": np.array(X_train_text), "X_test_text": np.array(X_test_text),
        "X_train_img": np.array(X_train_img), "X_test_img": np.array(X_test_img),
        "X_test_raw_text": X_test_raw_text,
        "y_train": y_train, "y_test": y_test
    }

def load_all_models():
    """
    Loads the pre-trained image, text, and multimodal models and their weights.

    Returns:
        A dictionary containing the loaded models.
    """
    print("Loading pre-trained models and weights...")
    img_model = load_model(IMG_MODEL_PATH)
    img_model.load_weights(IMG_WEIGHTS_PATH)

    text_model = load_model(TXT_MODEL_PATH)
    text_model.load_weights(TXT_WEIGHTS_PATH)

    merged_model = load_model(MRG_MODEL_PATH)
    merged_model.load_weights(MRG_WEIGHTS_PATH)

    return {"image": img_model, "text": text_model, "multimodal": merged_model}


def evaluate_models(models, data):
    """
    Calculates and prints the AUC for each model on the test set.

    Args:
        models: A dictionary of loaded Keras models.
        data: A dictionary of training and testing data splits.
    """
    print("\n--- Model Evaluation (AUC) ---")
    y_test = data['y_test']
    
    # Text Model
    pred_text = models['text'].predict(data['X_test_text'])
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_text, pos_label=1)
    print(f"Text Model AUC: {metrics.auc(fpr, tpr):.2f}")

    # Image Model
    pred_img = models['image'].predict(data['X_test_img'])
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_img, pos_label=1)
    print(f"Image Model AUC: {metrics.auc(fpr, tpr):.2f}")

    # Multimodal Model
    pred_merged = models['multimodal'].predict([data['X_test_img'], data['X_test_text']])
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_merged, pos_label=1)
    print(f"Multimodal AUC: {metrics.auc(fpr, tpr):.2f}")


def explain_image_with_gradients(image, image_model):
    """
    Uses Integrated Gradients to find and highlight anomalous regions in an image.

    Args:
        image: A single image array (e.g., a chest X-ray).
        image_model: The pre-trained image classification model.

    Returns:
        A tuple containing:
        - The original image with detected lesions circled.
        - The raw, full gradient map.
    """
    # Initialize the Integrated Gradients explainer
    ig = integrated_gradients(image_model)
    explanation_map = ig.explain(image, num_steps=50, a=-1) # Using baseline black image

    # Normalize gradients to a 0-1 range
    normalized_gradients = cv2.normalize(explanation_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    gray_gradients = cv2.cvtColor(normalized_gradients, cv2.COLOR_BGR2GRAY)

    # Threshold to keep only high-importance pixels
    thresholded_map = gray_gradients.copy()
    thresholded_map[thresholded_map < IG_GRADIENT_THRESHOLD] = 0

    # Find coordinates of high-importance pixels
    coords = np.argwhere(thresholded_map > 0)
    if coords.size == 0:
        print("No significant features found for clustering.")
        return image, gray_gradients

    # Cluster these pixels to find distinct regions (lesions)
    # The 'data' is structured as [[y1, x1], [y2, x2], ...]
    clusters = hcluster.fclusterdata(coords, LESION_CLUSTER_THRESHOLD, criterion="distance")
    
    # Create a DataFrame to analyze clusters
    df = pd.DataFrame(coords, columns=['y', 'x'])
    df['cluster'] = clusters
    
    # Filter out small, noisy clusters
    cluster_counts = df['cluster'].value_counts()
    relevant_clusters = cluster_counts[cluster_counts > 1].index

    output_image = image.copy()
    for cid in relevant_clusters:
        cluster_points = df[df['cluster'] == cid]
        
        # Calculate the center of the cluster
        x_center = int(cluster_points['x'].mean())
        y_center = int(cluster_points['y'].mean())
        
        # Calculate the maximum distance from the center to any point in the cluster
        distances = np.hypot(cluster_points['y'] - y_center, cluster_points['x'] - x_center)
        radius = max(distances) + LESION_CIRCLE_RADIUS_PADDING
        
        # Draw a circle around the detected region
        cv2.circle(output_image, (x_center, y_center), int(radius), (0, 0, 255), 2) # Red circle
        
    return output_image, gray_gradients


def explain_text_with_gradients(text_model, processed_text_input, raw_text_input):
    """
    Calculates word importance scores using gradients from the text model.
    Updated to use TensorFlow 2.x GradientTape.

    Args:
        text_model: The pre-trained text classification model.
        processed_text_input: The tokenized and padded input for the model.
        raw_text_input: The original string of the doctor's note.

    Returns:
        A Plotly annotated heatmap figure showing word importance.
    """
    # Ensure input is a single sample with a batch dimension
    if len(processed_text_input.shape) == 1:
        processed_text_input = np.expand_dims(processed_text_input, axis=0)

    # The layer to get gradients from (typically the embedding layer)
    embedding_layer = text_model.layers[1] # Assumes embedding is the second layer
    
    input_tensor = tf.convert_to_tensor(processed_text_input, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        # Get the output of the embedding layer
        embedding_output = embedding_layer(input_tensor)
        # Pass the embedding output through the rest of the model
        # Note: This requires a model that can be called on intermediate tensors.
        # If the model is purely sequential, this is straightforward.
        # Create a temporary model from the embedding output onwards.
        sub_model = tf.keras.Model(inputs=embedding_layer.output, outputs=text_model.output)
        prediction = sub_model(embedding_output)

    # Get gradients of the final prediction with respect to the embedding output
    gradients = tape.gradient(prediction, embedding_output)
    
    # Process gradients to get word-level importance
    word_gradients = gradients[0].numpy() # Get gradients for the first (only) sample
    word_importance_scores = np.sum(np.abs(word_gradients), axis=1) # Sum gradients across embedding dimension

    # Normalize scores for visualization
    min_val, max_val = np.min(word_importance_scores), np.max(word_importance_scores)
    normalized_scores = (word_importance_scores - min_val) / (max_val - min_val)
    
    # Align scores with original words
    words = raw_text_input.split()
    num_words = len(words)
    scores = list(normalized_scores[:num_words])
    
    # Create Plotly annotated heatmap
    # NOTE: Set up your Plotly credentials in your environment for online plotting.
    # e.g., os.environ['PLOTLY_USERNAME'] = 'your_user'
    # os.environ['PLOTLY_API_KEY'] = 'your_key'
    
    fig = ff.create_annotated_heatmap(
        z=[scores],
        x=words,
        annotation_text=[[f'{s:.2f}' for s in scores]],
        colorscale='Viridis',
        font_colors=['white', 'black'],
        hoverinfo='x+text'
    )
    fig.update_layout(title_text='<b>Word Importance Heatmap (Anomaly Detection)</b>')
    
    return fig


def main():
    """Main function to run the evaluation and explanation pipeline."""
    data = load_and_prepare_data()
    models = load_all_models()
    
    # --- 1. Model Evaluation ---
    evaluate_models(models, data)

    # --- 2. Anomaly Detection / Explainability ---
    print("\n--- Anomaly Detection Examples ---")

    # --- Image Example (Abnormal Patient) ---
    print(f"\nAnalyzing image for abnormal patient index: {ABNORMAL_PATIENT_IDX}")
    
    patient_img = data['X_test_img'][ABNORMAL_PATIENT_IDX]
    img_prediction = models['image'].predict(np.expand_dims(patient_img, axis=0))
    print(f"Image model confidence of abnormality: {img_prediction[0][0]:.4f}")

    explained_img, gradient_map = explain_image_with_gradients(patient_img, models['image'])

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(patient_img)
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')
    
    axes[1].imshow(gradient_map, cmap='hot')
    axes[1].set_title('Integrated Gradients Heatmap')
    axes[1].axis('off')

    axes[2].imshow(explained_img)
    axes[2].set_title('Detected Anomalous Region')
    axes[2].axis('off')
    
    plt.suptitle(f'Explainability for Patient Index {ABNORMAL_PATIENT_IDX}')
    plt.show()

    # --- Text Example (Normal Patient) ---
    print(f"\nAnalyzing text for normal patient index: {NORMAL_PATIENT_IDX}")
    
    patient_raw_text = data['X_test_raw_text'][NORMAL_PATIENT_IDX]
    patient_processed_text = data['X_test_text'][NORMAL_PATIENT_IDX]

    print("Original Text:\n", patient_raw_text)
    
    heatmap_fig = explain_text_with_gradients(
        models['text'],
        patient_processed_text,
        patient_raw_text
    )
    heatmap_fig.show()


if __name__ == '__main__':
    main()