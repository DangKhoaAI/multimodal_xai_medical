from sklearn import metrics
import numpy as np

def evaluate_models(models: dict, data: dict):
    y = data['y_test']

    # Text
    preds_text = models['text'].predict(data['X_test_text']).ravel()
    fpr, tpr, _ = metrics.roc_curve(y, preds_text, pos_label=1)
    auc_text = metrics.auc(fpr, tpr)
    # chuyển xác suất thành nhãn với threshold=0.5
    y_pred_text = (preds_text >= 0.5).astype(int)
    acc_text = metrics.accuracy_score(y, y_pred_text)
    print(f"Text Model → AUC: {auc_text:.2f}, Accuracy: {acc_text:.2f}")

    # Image
    preds_img = models['image'].predict(data['X_test_img']).ravel()
    fpr, tpr, _ = metrics.roc_curve(y, preds_img, pos_label=1)
    auc_img = metrics.auc(fpr, tpr)
    y_pred_img = (preds_img >= 0.5).astype(int)
    acc_img = metrics.accuracy_score(y, y_pred_img)
    print(f"Image Model → AUC: {auc_img:.2f}, Accuracy: {acc_img:.2f}")

    # Multimodal
    merged_preds = models['multimodal'].predict([
        data['X_test_text'],
        data['X_test_img']
    ]).ravel()
    fpr, tpr, _ = metrics.roc_curve(y, merged_preds, pos_label=1)
    auc_multi = metrics.auc(fpr, tpr)
    y_pred_multi = (merged_preds >= 0.5).astype(int)
    acc_multi = metrics.accuracy_score(y, y_pred_multi)
    print(f"Multimodal Model → AUC: {auc_multi:.2f}, Accuracy: {acc_multi:.2f}")
