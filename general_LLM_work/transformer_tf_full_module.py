# transformer_full.py
"""
Complete Transformer (Encoder + Decoder) implementation in TensorFlow/Keras.
Includes:
- Tokeniser (Keras TextVectorization)
- Positional Encoding
- Multi-Head Attention (custom and FlashAttention variant)
- Encoder Layer
- Decoder Layer
- Full Encoder-Decoder Model
- Training loop skeleton
- Einops version for attention projections
"""

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import numpy as np
from einops import rearrange

# Placeholder imports (for environments without TensorFlow)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError:
    print("Warning: TensorFlow is not installed. Using placeholders for classes and functions.")
    tf = type('tf', (), {})()
    keras = type('keras', (), {})()
    layers = type('layers', (), {})()

  # Placeholder Dense and Layer classes
    class Dense:
        def __init__(self, units, **kwargs):
            self.units = units
        def __call__(self, x):
            return x
    class Layer:
        def __init__(self): pass
        def __call__(self, *args, **kwargs): return args[0] if args else None

    layers.Dense = Dense
    layers.Layer = Layer
    layers.Dropout = lambda rate: (lambda x: x)
    layers.LayerNormalization = lambda epsilon=1e-6: (lambda x: x)
    
# -------------------------------------------------------
# Positional Encoding
# -------------------------------------------------------
def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)


# -------------------------------------------------------
# Scaled Dot-Product Attention (Custom)
# -------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scores /= tf.math.sqrt(dk)
    if mask is not None:
        scores += (mask * -1e9)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, V)


# -------------------------------------------------------
# Multi-Head Attention (Einops version)
# -------------------------------------------------------
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = layers.Dense(d_model)
        self.W_K = layers.Dense(d_model)
        self.W_V = layers.Dense(d_model)
        self.W_O = layers.Dense(d_model)

    def split_heads(self, x):
        return rearrange(x, "b t (h d) -> b h t d", h=self.num_heads)

    def combine_heads(self, x):
        return rearrange(x, "b h t d -> b t (h d)")

    def call(self, x_q, x_kv, mask=None):
        Q = self.split_heads(self.W_Q(x_q))
        K = self.split_heads(self.W_K(x_kv))
        V = self.split_heads(self.W_V(x_kv))

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        out = self.combine_heads(attn_out)
        return self.W_O(out)


# -------------------------------------------------------
# FlashAttention Variant using TF fused API
# -------------------------------------------------------
class FlashAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = layers.Dense(d_model)
        self.W_K = layers.Dense(d_model)
        self.W_V = layers.Dense(d_model)
        self.W_O = layers.Dense(d_model)

    def split_heads(self, x):
        return rearrange(x, "b t (h d) -> b h t d", h=self.num_heads)

    def combine_heads(self, x):
        return rearrange(x, "b h t d -> b t (h d)")

    def call(self, x_q, x_kv, mask=None):
        Q = self.split_heads(self.W_Q(x_q))
        K = self.split_heads(self.W_K(x_kv))
        V = self.split_heads(self.W_V(x_kv))

        attn = tf.nn.scaled_dot_product_attention(Q, K, V, attention_mask=mask)
        out = self.combine_heads(attn)
        return self.W_O(out)


# -------------------------------------------------------
# Feed Forward Network
# -------------------------------------------------------
class FeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        return self.seq(x)


# -------------------------------------------------------
# Encoder Layer
# -------------------------------------------------------
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask=None):
        attn = self.mha(x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# -------------------------------------------------------
# Decoder Layer
# -------------------------------------------------------
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, enc_out, lookahead_mask=None, padding_mask=None):
        self_attn = self.self_mha(x, x, lookahead_mask)
        x = self.norm1(x + self.dropout(self_attn))

        cross_attn = self.cross_mha(x, enc_out, padding_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        ffn_out = self.ffn(x)
        return self.norm3(x + self.dropout(ffn_out))


# -------------------------------------------------------
# Full Transformer Model
# -------------------------------------------------------
class Transformer(keras.Model):
    def __init__(self, vocab_size, max_len, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.d_model = d_model

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

        self.final_dense = layers.Dense(vocab_size)

    def add_positional(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def call(self, enc_inp, dec_inp, training=False):
        enc_x = self.add_positional(self.embedding(enc_inp))
        for layer in self.enc_layers:
            enc_x = layer(enc_x)

        dec_x = self.add_positional(self.embedding(dec_inp))
        for layer in self.dec_layers:
            dec_x = layer(dec_x, enc_x)

        return self.final_dense(dec_x)


# -------------------------------------------------------
# Tokeniser + Data Pipeline
# -------------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab_size=20000, seq_len=128):
        self.vectorizer = layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=seq_len)

    def adapt(self, text_ds):
        self.vectorizer.adapt(text_ds)

    def encode(self, text):
        return self.vectorizer(text)

    def decode(self, ids):
        vocab = self.vectorizer.get_vocabulary()
        inv = {i: t for i, t in enumerate(vocab)}
        return " ".join(inv[i] for i in ids if i < len(inv))


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def create_loss_fn():
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def loss(labels, logits):
        shift_labels = labels[:, 1:]
        shift_logits = logits[:, :-1]
        return loss_fn(shift_labels, shift_logits)
    return loss

@tf.function
def train_step(model, optimizer, x_enc, x_dec):
    with tf.GradientTape() as tape:
        logits = model(x_enc, x_dec, training=True)
        loss = create_loss_fn()(x_dec, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# -------------------------------------------------------
# Main build utility
# -------------------------------------------------------
def build_transformer(vocab_size=20000, max_len=128, d_model=256, num_heads=8, d_ff=1024, num_layers=6):
    return Transformer(vocab_size, max_len, d_model, num_heads, d_ff, num_layers)


# -------------------------------------------------------
# Example Usage (commented)
# -------------------------------------------------------
"""
text_data = tf.data.Dataset.from_tensor_slices([
    "Planet Mars", "transformers are powerful", "attention is all you need"
])

# Tokeniser
 tok = SimpleTokenizer()
 tok.adapt(text_data)

# Prepare sample encoded data
 enc = tok.encode(tf.constant(["hello world"]))
 dec = tok.encode(tf.constant(["transformers are powerful"]))

# Model
 model = build_transformer()
 optimizer = keras.optimizers.Adam()

# Training step
 loss = train_step(model, optimizer, enc, dec)
 print("Loss:", loss.numpy())
"""


# ==============================
# LoRA / QLoRA Implementation
# ==============================
# Lightweight Low-Rank Adaptation layer for attention projections.
# Replace Dense with LoRADense where fine-tuning.

class LoRADense(layers.Layer):
    def __init__(self, units, r=8, alpha=16, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dense = layers.Dense(units, use_bias=False)
        if r > 0:
            self.lora_A = self.add_weight(
                shape=(self.dense.kernel.shape[0], r),
                initializer="random_normal",
                trainable=True,
                name="lora_A"
            )
            self.lora_B = self.add_weight(
                shape=(r, units),
                initializer="zeros",
                trainable=True,
                name="lora_B"
            )
        else:
            self.lora_A, self.lora_B = None, None

    def call(self, inputs):
        base = self.dense(inputs)
        if self.r > 0:
            lora_out = tf.matmul(inputs, self.lora_A)
            lora_out = tf.matmul(lora_out, self.lora_B) * self.scaling
            return base + lora_out
        return base

# ==== QLoRA Adapter using 4-bit quantised base weights (simulated placeholder) ====
# Real QLoRA requires quantisation kernels; here we define structure only.

class QLoRADense(LoRADense):
    def build(self, input_shape):
        super().build(input_shape)
        # Placeholder quantised kernel (simulate int4 block)
        self.quant_kernel = tf.cast(self.dense.kernel, dtype=tf.int8)

    def call(self, inputs):
        # Dequantisation simulation (not real int4 computation)
        dq_kernel = tf.cast(self.quant_kernel, tf.float32)
        base = tf.matmul(inputs, dq_kernel)

        if self.r > 0:
            lora_out = tf.matmul(inputs, self.lora_A)
            lora_out = tf.matmul(lora_out, self.lora_B) * self.scaling
            return base + lora_out
        return base

# To enable LoRA in Transformer, replace Dense with LoRADense:
# Example:
# self.W_Q = LoRADense(d_k, r=8, alpha=32)
# self.W_K = LoRADense(d_k, r=8, alpha=32)
# self.W_V = LoRADense(d_k, r=8, alpha=32)

