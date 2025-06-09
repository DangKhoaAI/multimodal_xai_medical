import numpy as np
from keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten,
    concatenate, Embedding
)
from keras.models import Model
from keras.applications.densenet import DenseNet121
from keras import regularizers

def create_text_submodel(max_seq_length, vocab_size, embedding_dim,
                         filter_sizes, feature_maps, dropout_rate,
                         embedding_matrix=None):
    """
    Tạo submodel xử lý văn bản:
      - Input: chuỗi số nguyên có độ dài max_seq_length
      - Embedding: nếu bạn truyền embedding_matrix, hàm sẽ dùng pre-trained weights
      - N lần Conv1D + MaxPooling1D, rồi Flatten
      - Trả về tensor cuối cùng của text branch
    """
    text_input = Input(shape=(max_seq_length,), name='text_input')
    if embedding_matrix is not None:
        emb = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_seq_length,
            weights=[embedding_matrix],
            trainable=True,
            name='text_embedding'
        )(text_input)
    else:
        emb = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_seq_length,
            name='text_embedding'
        )(text_input)

    if dropout_rate:
        emb = Dropout(dropout_rate, name='text_embedding_dropout')(emb)

    conv_outputs = []
    for i, (filt, fmap) in enumerate(zip(filter_sizes, feature_maps)):
        x = Conv1D(
            filters=fmap,
            kernel_size=filt,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.03),
            name=f'conv1d_{i}'
        )(emb)
        x = MaxPooling1D(
            pool_size=2,
            strides=1,
            name=f'maxpool1d_{i}'
        )(x)
        x = Flatten(name=f'flatten_{i}')(x)
        conv_outputs.append(x)

    text_feat = concatenate(conv_outputs, name='text_concat')
    return text_input, text_feat


def create_image_submodel(img_target_size):
    """
    Tạo submodel xử lý ảnh:
      - Input: ảnh kích thước img_target_size
      - Base: DenseNet121 pretrained (ChestX-ray14 weights)
      - Lấy feature từ layer 'avg_pool', BatchNorm để ổn định
    """
    image_input = Input(shape=img_target_size, name='image_input')
    base = DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=image_input,
        classes=14,
        name='densenet121_base'
    )
    x = base.get_layer('avg_pool').output
    img_feat = BatchNormalization(name='img_batchnorm')(x)
    return image_input, img_feat


def build_multimodal_model(
    max_seq_length,
    vocab_size,
    embedding_dim,
    filter_sizes,
    feature_maps,
    dropout_rate,
    img_target_size,
    embedding_matrix=None
):
    # Text branch
    text_in, text_feat = create_text_submodel(
        max_seq_length, vocab_size, embedding_dim,
        filter_sizes, feature_maps, dropout_rate,
        embedding_matrix
    )

    # Image branch
    img_in, img_feat = create_image_submodel(img_target_size)

    # Fusion
    merged = concatenate([text_feat, img_feat], name='fusion_concat')
    x = BatchNormalization(name='fusion_batchnorm_1')(merged)
    x = Dense(512, activation='relu', name='fusion_dense')(x)
    x = Dropout(0.3, name='fusion_dropout')(x)
    x = BatchNormalization(name='fusion_batchnorm_2')(x)
    out = Dense(1, activation='softmax', name='output')(x)

    model = Model(inputs=[text_in, img_in], outputs=out, name='multimodal_model')
    return model
