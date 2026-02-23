import numpy as np

def setup_seed(seed):
    np.random.seed(seed)

setup_seed(30)

# Now import TensorFlow safely
import tensorflow as tf
tf.random.set_seed(30)  # Set seed after import
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

"""
This module defines the Har_Classifier model, which is a two-stream classifier for 
Human Activity Recognition (HAR) using accelerometer and gyroscope data. The model architecture 
includes convolutional layers for feature extraction, positional embeddings, multi-head self-attention, 
cross-attention between the two streams, and a classification head. The model is designed to classify activities into 13 classes.
"""
class PositionalEmbedding(layers.Layer):
    """Custom layer to add positional embeddings to tokens."""
    def __init__(self, max_pos, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=max_pos, output_dim=d_model)

    def call(self, x):
        n = tf.shape(x)[1]
        pos = tf.range(0, n)
        return x + self.pos_emb(pos)

"""
HAR Two-Stream Classifier without Patchify + Cross-Attention.
"""
class Har_Classifier(Model):

    def __init__(self,
                 input_shape=(500, 3),
                 ff_dim=32,
                 fconn_units=128,
                 dropout_rate=0.3,
                 d_model=128,
                 num_heads=4,
                 num_fusion_layers=1,
                 mlp_ratio=4,
                 max_pos=125,
                 num_classes=13,
                 **kwargs):

        # Store hyperparams as instance attributes
        self.ff_dim = ff_dim
        self.fconn_units = fconn_units
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers
        self.mlp_ratio = mlp_ratio
        self.max_pos = max_pos
        self.num_classes = num_classes

        # --- Helper functions (use self.* for params) ---
        def silu(x):
            return layers.Activation("swish")(x)

        def ffn_block(x, name_prefix):
            h = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln")(x)
            h = layers.Dense(self.mlp_ratio * self.d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f"{name_prefix}_ff1")(h)
            h = silu(h)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_ffdrop1")(h)
            h = layers.Dense(self.d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f"{name_prefix}_ff2")(h)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_ffdrop2")(h)
            return layers.Add(name=f"{name_prefix}_res")([x, h])

        def self_attn_block(x, name_prefix):
            h = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln")(x)
            h = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                name=f"{name_prefix}_mha"
            )(h, h)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(h)
            x = layers.Add(name=f"{name_prefix}_res")([x, h])
            x = ffn_block(x, name_prefix=f"{name_prefix}_ffn")
            return x

        def cross_attn_block(q_tokens, kv_tokens, name_prefix):
            qn = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lnq")(q_tokens)
            kvn = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lnkv")(kv_tokens)
            h = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                name=f"{name_prefix}_xattn"
            )(qn, kvn)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(h)
            x = layers.Add(name=f"{name_prefix}_res")([q_tokens, h])
            x = ffn_block(x, name_prefix=f"{name_prefix}_ffn")
            return x

        def stream_frontend(x, name_prefix):
            x = layers.Conv1D(filters=self.ff_dim, kernel_size=5, padding="same", name=f"{name_prefix}_conv1")(x)
            x = silu(x)
            x = layers.MaxPooling1D(pool_size=2, name=f"{name_prefix}_pool1")(x)
            x = layers.Dense(self.fconn_units, name=f"{name_prefix}_dense1")(x)
            x = silu(x)
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(x)
            x = layers.Conv1D(filters=self.ff_dim, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(x)
            x = silu(x)
            x = layers.MaxPooling1D(pool_size=2, name=f"{name_prefix}_pool2")(x)
            x = layers.Dense(self.fconn_units, name=f"{name_prefix}_dense2")(x)
            x = silu(x)
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop2")(x)
            return x

        def stream_tokens(x, name_prefix):
            tokens = layers.Dense(self.d_model, name=f"{name_prefix}_tokproj")(x)
            # FIX: Use custom layer to handle symbolic indexing and position embeddings
            tokens = PositionalEmbedding(self.max_pos, self.d_model, name=f"{name_prefix}_pos")(tokens)
            tokens = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_tokdrop")(tokens)
            return tokens

        # --- Build the graph ---
        accelerometer_input = layers.Input(shape=input_shape, name="accelerometer")
        gyroscope_input = layers.Input(shape=input_shape, name="gyroscope")

        acc_feat = stream_frontend(accelerometer_input, "acc")
        gyro_feat = stream_frontend(gyroscope_input, "gyro")

        acc_tok = stream_tokens(acc_feat, "acc")
        gyro_tok = stream_tokens(gyro_feat, "gyro")

        for i in range(self.num_fusion_layers):
            acc_tok = self_attn_block(acc_tok, name_prefix=f"f{i+1}_acc_sa")
            gyro_tok = self_attn_block(gyro_tok, name_prefix=f"f{i+1}_gyro_sa")
            acc_tok = cross_attn_block(acc_tok, gyro_tok, name_prefix=f"f{i+1}_acc_xattn")
            gyro_tok = cross_attn_block(gyro_tok, acc_tok, name_prefix=f"f{i+1}_gyro_xattn")

        z_acc = layers.GlobalAveragePooling1D(name="acc_pool")(acc_tok)
        z_gyro = layers.GlobalAveragePooling1D(name="gyro_pool")(gyro_tok)

        z = layers.Concatenate(name="latent_concat")([z_acc, z_gyro])
        z = layers.Dense(256, kernel_regularizer=l2(1e-4), name="latent_proj")(z)
        z = silu(z)
        z = layers.Dropout(self.dropout_rate, name="latent_drop")(z)

        # Classification head
        clf = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name="clf_dense1")(z)
        clf = layers.Dropout(self.dropout_rate, name="clf_drop")(clf)
        clf = layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(1e-4), name="clf_output")(clf)

        # Initialize Model
        super().__init__(
            inputs=[accelerometer_input, gyroscope_input],
            outputs=clf,
            name="HAR_TwoStream_Classifier",
            **kwargs
        )



"""
HAR SiLU Classifier with Patchify1D and Cross-Attention 
"""
class HarPatchifyClassifier2(models.Model):
    """
    HAR Two-Stream Patchify + Cross-Attention Classifier for 13 classes (Reduced Complexity).

    Modifications for reduced overfitting:
    - Lower d_model (128 -> 64)
    - Fewer num_heads (4 -> 2)
    - No fusion layers (num_fusion_layers=0)
    - Added BatchNormalization after dense layers
    - Higher dropout_rate (0.3 -> 0.4)

    Usage:
        model = HarPatchifyClassifier2()
        # model.summary()
        # model.fit([X_acc, X_gyro], y_labels, ...)
    """

    def __init__(self,
                 input_shape=(500, 3),
                 ff_dim=32,
                 fconn_units=128,
                 dropout_rate=0.4,  # Increased from 0.3
                 patch_length=10,
                 patch_stride=5,
                 d_model=64,  # Reduced from 128
                 num_heads=2,  # Reduced from 4
                 num_fusion_layers=0,  # Reduced from 1
                 mlp_ratio=4,
                 max_pos=1000,
                 num_classes=13,
                 **kwargs):

        # Store hyperparams as instance attributes
        self.ff_dim = ff_dim
        self.fconn_units = fconn_units
        self.dropout_rate = dropout_rate
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers
        self.mlp_ratio = mlp_ratio
        self.max_pos = max_pos
        self.num_classes = num_classes

        # --- Helper functions (use self.* for params) ---
        def silu(x):
            return layers.Activation("swish")(x)

        def ffn_block(x, name_prefix):
            h = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln")(x)
            h = layers.Dense(self.mlp_ratio * self.d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f"{name_prefix}_ff1")(h)
            h = silu(h)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_ffdrop1")(h)
            h = layers.Dense(self.d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f"{name_prefix}_ff2")(h)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_ffdrop2")(h)
            return layers.Add(name=f"{name_prefix}_res")([x, h])

        def self_attn_block(x, name_prefix):
            h = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln")(x)
            h = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                name=f"{name_prefix}_mha"
            )(h, h)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(h)
            x = layers.Add(name=f"{name_prefix}_res")([x, h])
            x = ffn_block(x, name_prefix=f"{name_prefix}_ffn")
            return x

        def cross_attn_block(q_tokens, kv_tokens, name_prefix):
            qn = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lnq")(q_tokens)
            kvn = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lnkv")(kv_tokens)
            h = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                name=f"{name_prefix}_xattn"
            )(qn, kvn)
            h = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(h)
            x = layers.Add(name=f"{name_prefix}_res")([q_tokens, h])
            x = ffn_block(x, name_prefix=f"{name_prefix}_ffn")
            return x

        def stream_frontend(x, name_prefix):
            x = layers.Conv1D(filters=self.ff_dim, kernel_size=5, padding="same", name=f"{name_prefix}_conv1")(x)
            x = silu(x)
            x = layers.MaxPooling1D(pool_size=2, name=f"{name_prefix}_pool1")(x)
            x = layers.Dense(self.fconn_units, name=f"{name_prefix}_dense1")(x)
            x = silu(x)
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(x)
            x = layers.Conv1D(filters=self.ff_dim, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(x)
            x = silu(x)
            x = layers.MaxPooling1D(pool_size=2, name=f"{name_prefix}_pool2")(x)
            x = layers.Dense(self.fconn_units, name=f"{name_prefix}_dense2")(x)
            x = silu(x)
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop2")(x)
            return x

        def stream_patch_tokens(x, name_prefix):
            patches = Patchify1D(self.patch_length, self.patch_stride, name=f"{name_prefix}_patchify")(x)
            tokens = layers.Dense(self.d_model, name=f"{name_prefix}_tokproj")(patches)
            # FIX: Use custom layer to handle symbolic indexing and position embeddings
            tokens = PositionalEmbedding(self.max_pos, self.d_model, name=f"{name_prefix}_pos")(tokens)
            tokens = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_tokdrop")(tokens)
            return tokens

        # --- Build the graph ---
        accelerometer_input = layers.Input(shape=input_shape, name="accelerometer")
        gyroscope_input = layers.Input(shape=input_shape, name="gyroscope")

        acc_feat = stream_frontend(accelerometer_input, "acc")
        gyro_feat = stream_frontend(gyroscope_input, "gyro")

        acc_tok = stream_patch_tokens(acc_feat, "acc")
        gyro_tok = stream_patch_tokens(gyro_feat, "gyro")

        for i in range(self.num_fusion_layers):
            acc_tok = self_attn_block(acc_tok, name_prefix=f"f{i+1}_acc_sa")
            gyro_tok = self_attn_block(gyro_tok, name_prefix=f"f{i+1}_gyro_sa")
            acc_tok = cross_attn_block(acc_tok, gyro_tok, name_prefix=f"f{i+1}_acc_xattn")
            gyro_tok = cross_attn_block(gyro_tok, acc_tok, name_prefix=f"f{i+1}_gyro_xattn")

        z_acc = layers.GlobalAveragePooling1D(name="acc_pool")(acc_tok)
        z_gyro = layers.GlobalAveragePooling1D(name="gyro_pool")(gyro_tok)

        z = layers.Concatenate(name="latent_concat")([z_acc, z_gyro])
        z = layers.Dense(256, kernel_regularizer=l2(1e-4), name="latent_proj")(z)
        z = silu(z)
        z = layers.BatchNormalization(name="latent_bn")(z)  # Added BatchNormalization
        z = layers.Dropout(self.dropout_rate, name="latent_drop")(z)

        # Classification head
        clf = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name="clf_dense1")(z)
        clf = layers.BatchNormalization(name="clf_bn")(clf)  # Added BatchNormalization
        clf = layers.Dropout(self.dropout_rate, name="clf_drop")(clf)
        clf = layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(1e-4), name="clf_output")(clf)

        # Initialize Model
        super().__init__(
            inputs=[accelerometer_input, gyroscope_input],
            outputs=clf,
            name="HAR_TwoStream_Patchify_Classifier2",
            **kwargs
        )