import tensorflow as tf
# from mobilevit.models.mha import MHAttention
from tensorflow.keras import layers


def mlp(input_layer: layers.Input, hidden_units: int, dropout:bool, dropout_rate: int, name: str):
    """
    MLP layer.

    Args:
        input_layer: input tensor.
        hidden_units (int): list of hidden units.
        dropout_rate (int): dropout rate.

    Returns:
        output tensor of the MLP layer.
    """
    x = layers.Dense(hidden_units, activation=tf.nn.swish, name=name+f"Dense_{hidden_units}")(input_layer)
    if dropout:
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(
    x: layers.Input, transformer_layers: int, projection_dim: int, num_heads: int = 2, name: str = 'transformer_',
):
    """
    Transformer block.

    Args:
        x: input tensor.
        transformer_layers (int): number of transformer layers.
        projection_dim (int): projection dimension.
        num_heads (int): number of heads.

    Returns:
        output tensor of the transformer block.
    """
    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-05, name=name+f"{i+2}_attn_ln")(x)
        
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, name=name+f"{i+2}_attn_mha")(x1, x1)
        
        x = layers.Add()([attention_output, x])
        
        x2 = layers.LayerNormalization(epsilon=1e-05, name=name+f"{i+2}_mlp_ln")(x)
    
        x2 = mlp(
            x2,
            hidden_units=projection_dim * 2,
            dropout=False,
            dropout_rate=0.1,
            name=name+f"{i+2}_mlp_"
        )

        x2 = mlp(
            x2,
            hidden_units=projection_dim,
            dropout=True,
            dropout_rate=0.1,
            name=name+f"{i+2}_mlp_"
        )
       
        x = layers.Add(name=name+f"{i+2}_output")([x2, x])

    return x
