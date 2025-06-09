from keras.models import load_model
from config import IMG_MODEL_PATH, TXT_MODEL_PATH, MRG_MODEL_PATH

def load_all_models():
    img_model     = load_model(IMG_MODEL_PATH, compile=False)
    text_model    = load_model(TXT_MODEL_PATH, compile=False)
    multimodal    = load_model(MRG_MODEL_PATH, compile=False)
    return {'image': img_model, 'text': text_model, 'multimodal': multimodal}
