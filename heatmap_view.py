import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from flask import jsonify
import logging

logger = logging.getLogger(__name__)

def get_attention_data(attention_path, story_id):
    """
    Load attention data for a given story ID.
    This function reads encoder, decoder, and cross-attention tensors and token data.
    Returns a tuple with attention data and tokens if successful, otherwise returns None.
    """
    attention_dir = attention_path / str(story_id)
    logger.info(f"Loading attention data from {attention_dir}")

    if not attention_dir.exists():
        logger.error(f"Attention directory does not exist: {attention_dir}")
        return None

    try:
        encoder_attentions = [np.load(attention_dir / f'encoder_attentions_layer_{i}.npy') for i in range(12)]
        logger.info(f"Loaded encoder attentions for layers 0-11")
        decoder_attentions = [np.load(attention_dir / f'decoder_attentions_layer_{i}.npy') for i in range(12)]
        logger.info(f"Loaded decoder attentions for layers 0-11")
        cross_attentions = [np.load(attention_dir / f'cross_attentions_layer_{i}.npy') for i in range(12)]
        logger.info(f"Loaded cross attentions for layers 0-11")
    except Exception as e:
        logger.error(f"Error loading attention arrays: {e}")
        return None

    try:
        with open(attention_dir / "tokens.json") as f:
            tokens = json.load(f)
            logger.info("Loaded tokens.json")
    except Exception as e:
        logger.error(f"Error loading tokens.json: {e}")
        return None

    encoder_text = tokens.get('encoder_text', [])
    generated_text = tokens.get('generated_text', "")
    generated_text_tokens = tokens.get('generated_text_tokens', [])

    logger.info("Loaded encoder_text: %s", encoder_text)
    logger.info("Loaded generated_text: %s", generated_text)
    logger.info("Loaded generated_text_tokens: %s", generated_text_tokens)

    return encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens

def plot_attention_heatmap(attention, x_tokens, y_tokens, title, image_path):
    """
    Plot and save an attention heatmap.

    Parameters:
    attention (numpy.ndarray): The attention weights to be visualized.
    x_tokens (list of str): The input tokens.
    y_tokens (list of str): The generated text tokens.
    title (str): The title for the heatmap.
    image_path (str): The path to save the generated heatmap image.
    """
    logger.info("Number of x_tokens (input): %d", len(x_tokens))
    logger.info("Number of y_tokens (generated text): %d", len(y_tokens))
    logger.info("Attention matrix shape: %s", attention.shape)

    if attention.shape[-1] != len(x_tokens) or attention.shape[-2] != len(y_tokens):
        logger.error("Attention dimensions do not match the token list dimensions.")
        return

    fig_width = max(15, len(x_tokens) / 2)
    fig_height = max(10, len(y_tokens) / 2)

    plt.figure(figsize=(fig_width, fig_height))
    logger.info("Attention matrix shape for plotting: %s", attention.shape)
    logger.info("Number of input tokens: %d", len(x_tokens))
    logger.info("Number of output tokens: %d", len(y_tokens))

    sns.heatmap(attention, xticklabels=x_tokens, yticklabels=y_tokens, cmap='viridis', cbar=True)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Input Tokens', fontsize=12)
    plt.ylabel('Generated Text Tokens', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the directory exists
    logger.info(f"Saving heatmap to {image_path}")
    plt.savefig(image_path)
    plt.close()

def visualize_attention(request, data_dir, load_data_func, logger, plot_heatmap_func):
    """
    Generate and return the attention heatmap for a specific story.
    """
    model_key = request.json.get('model_key')
    story_index = request.json.get('story_index')
    if model_key is None or story_index is None:
        return jsonify({"error": "Model key or Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data_path = data_dir / model_key / 'test_data_sample-attention.csv'
    data = load_data_func(data_path)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    story_id = data.iloc[story_index]["StoryID"]

    attention_path = data_dir / model_key / 'attentions'
    image_path = generate_attention_image_path(model_key, story_id, data_dir)

    try:
        result = get_attention_data(attention_path, story_id)
        if result is None:
            return jsonify({"error": "Error loading attention data"}), 500
        encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens = result
        logger.info("Attention data loaded for story index %d", story_index)
        logger.info("Generated Text Tokens: %s", generated_text_tokens)
    except Exception as e:
        logger.error("Error loading attention data: %s", str(e))
        return jsonify({"error": str(e)}), 500

    try:
        first_layer_attention = cross_attentions[0]
        if isinstance(first_layer_attention, tuple):
            first_layer_attention = first_layer_attention[0]
        first_batch_attention = first_layer_attention[0]
        logger.info("Shape of first batch attention: %s", first_batch_attention.shape)

        if first_batch_attention.ndim == 3:
            attention_to_plot = first_batch_attention.mean(axis=0)
            logger.info("Averaged attention shape: %s", attention_to_plot.shape)
        elif first_batch_attention.ndim == 2:
            attention_to_plot = first_batch_attention
        else:
            logger.error("Unexpected attention matrix dimension: %dD", first_batch_attention.ndim)
            raise ValueError(f"Unexpected attention matrix dimension: {first_batch_attention.ndim}D")

        plot_heatmap_func(attention_to_plot, encoder_text, generated_text_tokens, "Cross-Attention Weights (First Layer)", image_path)
        logger.info(f"Generated heatmap image path: {image_path}")  # Debug: Print generated image path
    except Exception as e:
        logger.error("Error generating heatmap: %s", str(e))
        return jsonify({"error": str(e)}), 500

    return jsonify({"image_path": str(image_path)})

def generate_attention_image_path(model_key, story_id, base_dir):
    """
    Generate the path for the attention heatmap image based on model key and story ID.

    Parameters:
    model_key (str): The model identifier.
    story_id (str): The story identifier.
    base_dir (Path): The base directory where the data is stored.

    Returns:
    str: The path to the attention heatmap image.
    """
    return base_dir / model_key / 'attentions' / story_id / f'attention_heatmap_{story_id}.png'
