import tensorflow as tf
from tensorflow.keras import layers

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = layers.Dense(d_k)
        self.W_K = layers.Dense(d_k)
        self.W_V = layers.Dense(d_k)
        self.scale = tf.math.sqrt(tf.cast(d_k, tf.float32))

    def call(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_Q(x)  # (batch, seq_len, d_k)
        K = self.W_K(x)  # (batch, seq_len, d_k)
        V = self.W_V(x)  # (batch, seq_len, d_k)

        # Attention scores
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)

        # Output = weights * V
        output = tf.matmul(weights, V)
        return output, weights


# Example usage
batch, seq_len, d_model, d_k = 2, 5, 64, 32
x = tf.random.normal((batch, seq_len, d_model))

attention = ScaledDotProductAttention(d_model, d_k)
out, attn_weights = attention(x)

print("Output shape:", out.shape)
print("Attention weights shape:", attn_weights.shape)
