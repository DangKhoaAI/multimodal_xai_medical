# Medical Multimodal with Transfer Learning

This project focuses on leveraging deep learning techniques for medical data analysis, incorporating both image and text modalities. It appears to involve tasks such as disease detection or classification using Convolutional Neural Networks (CNNs) and transfer learning.

## Project Structure

The project is organized into several key components:

-   `data/`: Contains the dataset, including image IDs, raw text, and labels (`ids_raw_texts_labels.csv`), processed text data (`text_processed.pkl`), and vocabulary (`vocab.json`).
-   `image_preprocessing/`: Includes notebooks or scripts for preprocessing image data.
-   `text_preprocessing/`: Includes notebooks or scripts for preprocessing text data.
-   `model/`: Stores trained model weights and architectures (e.g., `img_model_final.h5`, `text_model_final.h5`, `full_weights_best_final.hdf5`).
-   `*.py` files:
    -   `single_model.py`: Script for training individual image and text models.
    -   `cnn_model.py`: Defines the CNN architecture used for text processing.
    -   `multimodaling.py`: Likely handles the combination of image and text modalities.
    -   `evaluation_detection.py`: Script for evaluating model performance.
    -   `explaining.py` & `explainingrefactor.py`: Scripts related to model explainability (e.g., using LIME or SHAP).
    -   `IntegratedGradients.py`: Implementation or usage of Integrated Gradients for model interpretation.
-   `*.ipynb` files: Jupyter notebooks for experimentation, evaluation (`evaluation_detection.ipynb`), explainability (`explaining.ipynb`), multimodal analysis (`multimodaling.ipynb`), and single model training/analysis (`single_model.ipynb`).
-   `requirements.txt`: Lists project dependencies.
-   `repo_images/`: Contains images used likely for documentation or visualization within notebooks.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd medical-multimodal-with-transfer-learning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Based on the `requirements.txt` file, you'll need `numpy` and `keras`.
    You will also likely need `tensorflow`, `pandas`, `scikit-learn`, and `nltk`. Install them if not already covered:
    ```bash
    pip install tensorflow pandas scikit-learn nltk
    ```
    Download NLTK data if you haven't already:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

4.  **Download GloVe embeddings:**
    The `single_model.py` script requires GloVe embeddings (`glove.300d.txt`). You will need to download this file and update the `GLOVE_PATH` variable in `single_model.py` accordingly.

5.  **Dataset:**
    -   Ensure the image data is available in the directory specified by `IMAGE_DIR` (default: `../NLMCXR_png/`) in `single_model.py`.
    -   Ensure the labels CSV file is available at the path specified by `LABEL_CSV` (default: `../ids_texts_labels.csv`) in `single_model.py`. The provided `data/ids_raw_texts_labels.csv` might be this file, ensure the path in the script matches its location relative to the script.

## Running the Code

### Training Models

The `single_model.py` script is used to train the image and text models.

1.  **Configure Parameters:**
    Open `single_model.py` and adjust the configuration parameters at the top of the file as needed:
    -   `IMAGE_DIR`: Path to the directory containing PNG images.
    -   `LABEL_CSV`: Path to the CSV file containing IDs, texts, and labels.
    -   `GLOVE_PATH`: Path to the downloaded GloVe word embeddings file.
    -   Other parameters like `MAX_NUM_WORDS`, `EMBEDDING_DIM`, `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `NB_EPOCHS`, etc.

2.  **Run the training script:**
    ```bash
    python single_model.py
    ```
    This will:
    -   Load and preprocess image data.
    -   Build and train an image model (DenseNet121 based).
    -   Load and preprocess text data.
    -   Build and train a text CNN model (optionally using GloVe embeddings).
    -   Save the best text model weights to `text_model.h5`.

### Evaluation

The `evaluation_detection.ipynb` notebook and `evaluation_detection.py` script are likely used for evaluating the performance of the trained models (e.g., calculating AUC). You would typically load the saved models and run predictions on a test set.

The notebook `evaluation_detection.ipynb` loads models from a `best_models/` directory. You might need to create this directory and move your best performing saved models there (e.g., `img_model_final.h5`, `text_model_final.h5`, `full_model_final.h5`) along with their corresponding weights files.

### Multimodal Analysis

The `multimodaling.ipynb` notebook and `multimodaling.py` script are likely used for combining the features or predictions from the individual image and text models to create a multimodal model.

### Explainability

The `explaining.ipynb` notebook and `explaining.py` script, along with `IntegratedGradients.py`, are used for interpreting model predictions and understanding which features (e.g., words in text, regions in images) are most influential.

## Models

The project involves building and training:

-   **Image Model:** A CNN based on DenseNet121, pre-trained on ImageNet, and fine-tuned for the medical imaging task.
-   **Text Model:** A CNN model for text classification, potentially using pre-trained GloVe word embeddings.
-   **Multimodal Model:** (Inferred from `multimodaling.ipynb` and `full_model_final.h5`) A model that combines information from both image and text modalities.

## Contributing

Please refer to the existing coding style and add tests for any new features.

## License

(Specify License if applicable - e.g., MIT, Apache 2.0)
