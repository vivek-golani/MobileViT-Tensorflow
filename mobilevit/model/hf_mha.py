import tensorflow as tf
from tensorflow.keras import layers

class MobileViTConfig:
    def __init__(
        self,
        num_attention_heads=4,
        mlp_ratio=2.0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        qkv_bias=True,
    ):
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.qkv_bias = qkv_bias

class TFMobileViTSelfAttention(layers.Layer):
    def __init__(self, config, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)

        self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="query")
        self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="key")
        self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="value")

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(hidden_states)[0]

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores + 1e-9, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, shape=(batch_size, -1, self.all_head_size))
        return context_layer


class TFMobileViTSelfOutput(layers.Layer):
    def __init__(self, config, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states


class TFMobileAttention(layers.Layer):
    def __init__(self, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        config  = MobileViTConfig()
        self.attention = TFMobileViTSelfAttention(config, hidden_size, name="attention")
        self.dense_output = TFMobileViTSelfOutput(config, hidden_size, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        self_outputs = self.attention(hidden_states, training=training)
        attention_output = self.dense_output(self_outputs, training=training)
        return attention_output