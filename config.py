# config.py
DATA_DIR               = 'data/'
MODEL_DIR              = 'checkpoints/'
TEXT_PROCESSED_PATH    = DATA_DIR + 'text_processed.pkl'
IMG_PROCESSED_PATH     = DATA_DIR + 'x_ray_processed.pkl'
ORIGINAL_DATA_PATH     = DATA_DIR + 'ids_raw_texts_labels.csv'

IMG_MODEL_PATH         = MODEL_DIR + 'img_model_final.h5'
TXT_MODEL_PATH         = MODEL_DIR + 'text_model_final.h5'
MRG_MODEL_PATH         = MODEL_DIR + 'multi_model.h5'

TEST_SET_SIZE          = 0.2
RANDOM_STATE           = 42
# Kích thước đầu vào cho ảnh (width, height)
IMG_TARGET_SIZE = (224, 224)

# Explainability parameters
IG_GRADIENT_THRESHOLD      = 0.70
LESION_CLUSTER_THRESHOLD   = 10
LESION_CIRCLE_RADIUS_PAD    = 5
ABNORMAL_PATIENT_IDX        = 125
NORMAL_PATIENT_IDX          = 120

# Text preproc
MAX_NUM_WORDS  = 15000
MAX_SEQ_LENGTH = 140
