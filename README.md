    # Medical Multimodal with Transfer Learning

    This project focuses on leveraging deep learning techniques for medical data analysis, incorporating both image and text modalities. It includes a Flask web application for model inference and explainability.

    ## Project Structure

    The project is organized into several key components:

    -   `app.py`: Main Flask web application for interacting with the models, performing inference, and visualizing explanations.
    -   `config.py`: Configuration file for managing paths, model parameters, and other settings.
    -   `data/`: Contains the dataset, including image IDs, raw text, and labels (`ids_raw_texts_labels.csv`), processed text data (`text_processed.pkl`), and vocabulary (`vocab.json`).
    -   `image_preprocessing/`: Includes notebooks or scripts for preprocessing image data. (Note: Content not verified in this update)
    -   `text_preprocessing/`: Includes notebooks or scripts for preprocessing text data. (Note: Content not verified in this update)
    -   `models/`: Contains scripts for model definitions (`cnn_model.py`, `multimodaling.py`, `multimodalingRE.py`), model loading (`load_models.py`), and training (`single_model.py`). Trained model weights are typically stored in `checkpoints/`.
    -   `checkpoints/`: Stores trained model weights (e.g., `img_model_final.h5`, `text_model_final.h5`, `multi_model.h5`).
    -   `evaluation/`: Contains scripts and notebooks for evaluating model performance (e.g., `evaluate.py`, `evaluation_detection.ipynb`).
    -   `explainability/`: Contains scripts for model explainability, including `IntegratedGradients.py`, `image_explainer.py`, and `text_explainer.py`.
    -   `explainingRE.py`: Script for model evaluation and generating feature-level explanations.
    -   `static/`: Contains static assets for the web application (CSS, images).
    -   `templates/`: Contains HTML templates for the web application.
    -   `*.ipynb` files: Jupyter notebooks for experimentation, evaluation (`evaluation/evaluation_detection.ipynb`), explainability, multimodal analysis (`models/multimodaling.ipynb`), and single model training/analysis (`models/single_model.ipynb`).
    -   `requirements.txt`: Lists project dependencies.

    ## Setup and Installation

    1.  **Clone the repository:**
        ```bash
        git clone https://github.com/DangKhoaAI/multimodal_xai_medical
        cd medical-multimodal-with-transfer-learning
        ```

    2.  **Create a virtual environment (recommended):**
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
        ```

    3.  **Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
        Ensure all dependencies from `requirements.txt` are installed. You may also need to manually install `tensorflow`, `pandas`, `scikit-learn`, and `nltk` if they are not listed or if issues arise.
        ```bash
        pip install tensorflow pandas scikit-learn nltk
        ```
        Download NLTK data if you haven't already:
        ```python
        import nltk
        nltk.download('stopwords')
        ```

    4.  **Download GloVe embeddings (if applicable):**
        If text models use GloVe embeddings (e.g., as referenced in `single_model.py`), you will need to download the appropriate GloVe file (e.g., `glove.6B.300d.txt`) and update the path in the relevant scripts or configuration (e.g., `GLOVE_PATH` in `single_model.py` or a similar variable in `config.py` if text preprocessing relies on it).

    5.  **Dataset and Model Paths:**
        -   Ensure the image data is available in the directory specified in `config.py` (e.g., `DATA_DIR`).
        -   Ensure the labels CSV file (`ids_raw_texts_labels.csv`) is available at the path specified in `config.py` (e.g., `ORIGINAL_DATA_PATH`).
        -   Verify that model paths in `config.py` (e.g., `IMG_MODEL_PATH`, `TXT_MODEL_PATH`, `MRG_MODEL_PATH`) point to the correct trained model files in the `checkpoints/` directory.

    ## Running the Code

    ### 1. Running the Web Application

    The primary way to interact with the models is through the Flask web application.

    1.  **Ensure Configuration:**
        Verify all paths in `config.py` are correctly set up (dataset, model checkpoints).

    2.  **Run the Flask app:**
        ```bash
        python app.py
        ```
        This will start a local development server. Open your web browser and navigate to the provided URL (usually `http://127.0.0.1:5000/`). The application allows you to input text and/or upload images for prediction and to see model explanations.

    ### 2. Training Models (Optional)

    The `models/single_model.py` script can be used to train the individual image and text models if you need to retrain them or train them on new data.

    1.  **Configure Parameters:**
        Open `models/single_model.py` and adjust configuration parameters as needed. Some of these might also be managed or referenced in `config.py`. Key parameters include:
        -   Paths to image directory, labels CSV, and GloVe embeddings.
        -   Hyperparameters like `MAX_NUM_WORDS`, `EMBEDDING_DIM`, `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `NB_EPOCHS`, etc.

    2.  **Run the training script:**
        ```bash
        python models/single_model.py
        ```
        This will typically:
        -   Load and preprocess image data.
        -   Build and train an image model.
        -   Load and preprocess text data.
        -   Build and train a text model.
        -   Save the best model weights (e.g., to the `checkpoints/` directory, ensure paths align with `config.py`).

    ### 3. Evaluation

    Scripts and notebooks in the `evaluation/` directory (e.g., `evaluate.py`, `evaluation_detection.ipynb`) and `explainingRE.py` can be used for evaluating model performance. You would typically load the saved models (paths from `config.py`) and run predictions on a test set.

    ### 4. Multimodal Analysis

    The `models/multimodaling.ipynb` notebook and `models/multimodaling.py` or `models/multimodalingRE.py` scripts are used for combining features or predictions from individual image and text models to create and potentially train a multimodal model.

    ### 5. Explainability

    The `explainability/` scripts (`image_explainer.py`, `text_explainer.py`, `IntegratedGradients.py`) are used by `app.py` to generate explanations. `explainingRE.py` also contains functions for generating explanations and can be run independently for analysis.

    ## Models

    The project involves building, training, and using:

    -   **Image Model:** A CNN (e.g., based on DenseNet121), potentially pre-trained and fine-tuned for medical imaging.
    -   **Text Model:** A CNN or other neural network model for text classification, possibly using pre-trained word embeddings like GloVe.
    -   **Multimodal Model:** A model that combines information from both image and text modalities, whose weights are stored (e.g. `multi_model.h5` in `checkpoints/`).
