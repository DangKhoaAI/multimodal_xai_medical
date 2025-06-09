import numpy as np
import tensorflow as tf
import plotly.figure_factory as ff

def explain_text_with_gradients(text_model, processed_input, raw_text: str) -> str:
    if processed_input.ndim == 1:
        processed_input = np.expand_dims(processed_input, 0)
    embedding = text_model.layers[1]
    inp = tf.convert_to_tensor(processed_input, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        emb_out = embedding(inp)
        sub = tf.keras.Model(embedding.output, text_model.output)
        pred = sub(emb_out)
    grads = tape.gradient(pred, emb_out)[0].numpy()
    scores = np.sum(np.abs(grads), axis=1)
    # chuẩn hoá
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    words = raw_text.split()
    scores = scores[:len(words)]

    fig = ff.create_annotated_heatmap(
        z=[scores], x=words,
        annotation_text=[[f"{s:.2f}" for s in scores]],
        colorscale='Viridis', font_colors=['white','black']
    )
    fig.update_layout(title_text='Word Importance Heatmap')

    # Trả về HTML fragment, không show() browser
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
