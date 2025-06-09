# scripts/run_pipeline.py
from data.loader import load_and_prepare_data
from models.load_models import load_all_models
from evaluation.evaluate import evaluate_models
from explainability.image_explainer import explain_image_with_gradients
from explainability.text_explainer import explain_text_with_gradients
from config import (
    TEXT_PROCESSED_PATH,
    IMG_PROCESSED_PATH,
    ORIGINAL_DATA_PATH,
    TEST_SET_SIZE,
    RANDOM_STATE,
    ABNORMAL_PATIENT_IDX,
    NORMAL_PATIENT_IDX,
)

def main():
    data = load_and_prepare_data(
        TEXT_PROCESSED_PATH,
        IMG_PROCESSED_PATH,
        ORIGINAL_DATA_PATH,
        TEST_SET_SIZE,
        RANDOM_STATE
    )
    models = load_all_models()
    evaluate_models(models, data)
    # ví dụ hình ảnh
    #img, heat = explain_image_with_gradients(
    #    data['X_test_img'][ABNORMAL_PATIENT_IDX], models['image'])
    # vẽ matplotlib hoặc xuất file…
    # ví dụ text
    fig = explain_text_with_gradients(
        models['text'], data['X_test_text'][NORMAL_PATIENT_IDX],
        data['X_test_raw_text'][NORMAL_PATIENT_IDX])
    fig.show()

if __name__=='__main__':
    main()
