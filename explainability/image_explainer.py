import numpy as np
import cv2
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.utils import normalize


def explain_image_with_gradients(image: np.ndarray, model) -> tuple:
    """
    Generate a refined Grad-CAM overlay with contour highlighting for a preprocessed input image.

    Returns:
        overlay (np.ndarray): Original image with heatmap and contour overlays (uint8 RGB).
        heatmap (np.ndarray): Normalized heatmap (H, W), values in [0, 1].
    """
    # 1. Build Grad-CAM
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
    def positive_score(output):
        return output[:, 0]
    cam = gradcam(positive_score, np.expand_dims(image, 0))[0]

    # 2. Normalize and apply gamma for contrast
    heatmap = normalize(cam)
    gamma = 1.2
    heatmap_adj = np.power(heatmap, gamma)

    # 3. Recover original image
    orig_img = np.uint8(np.clip((image + 1.0) * 127.5, 0, 255))

    # 4. Create colored heatmap with cooler colormap
    heatmap_uint8 = np.uint8(255 * heatmap_adj)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 5. Soft overlay for context
    alpha_hm = 0.5
    overlay = cv2.addWeighted(orig_img, 1 - alpha_hm, heatmap_color, alpha_hm, 0)

    # 6. Optional: contour highlight of the hottest regions
    #    threshold to extract top 30% activations
    thresh_val = np.percentile(heatmap_adj, 70)
    mask = (heatmap_adj >= thresh_val).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = overlay.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # ignore small areas
            continue
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)

    # Return contour version for clarity
    return contour_img, heatmap
