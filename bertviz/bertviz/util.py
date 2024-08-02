import numpy as np

def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for i, layer_attention in enumerate(attention):
        # Log the shape of each layer's attention tensor
        print(f"Layer {i} attention shape before squeeze: {layer_attention.shape}")
        # Ensure the tensor is 4D (batch_size, num_heads, seq_len, seq_len)
        if len(layer_attention.shape) != 4:
            raise ValueError(f"Layer {i} attention tensor does not have the correct number of dimensions. Expected 4D, got {len(layer_attention.shape)}D")
        # layer_attention = layer_attention.squeeze(0)  # Remove batch dimension
        print(f"Layer {i} attention shape after squeeze: {layer_attention.shape}")
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    stacked_attention = np.stack(squeezed)
    print(f"Stacked attention shape: {stacked_attention.shape}")
    return stacked_attention

def num_layers(attention):
    return len(attention)

def num_heads(attention):
    return attention[0].shape[1]  # Assuming attention is a list of numpy arrays with shape (batch_size, num_heads, seq_len, seq_len)

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]
