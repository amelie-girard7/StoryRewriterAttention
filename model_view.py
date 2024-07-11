from flask import jsonify, make_response
from bertviz.bertviz.model_view import model_view
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
        decoder_attentions = [np.load(attention_dir / f'decoder_attentions_layer_{i}.npy') for i in range(12)]
        cross_attentions = [np.load(attention_dir / f'cross_attentions_layer_{i}.npy') for i in range(12)]
    except Exception as e:
        logger.error(f"Error loading attention arrays: {e}")
        return None

    try:
        with open(attention_dir / "tokens.json") as f:
            tokens = json.load(f)
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

def visualize_model_view(request, data_dir, load_data_func, logger):
    """
    Generate and return the model view visualization for the attention mechanism.
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

    try:
        result = get_attention_data(attention_path, story_id)
        if result is None:
            return jsonify({"error": "Error loading attention data"}), 500
        encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens = result
        logger.info("Attention data loaded for story index %d", story_index)
        logger.info("Generated Text Tokens: %s", generated_text_tokens)
        logger.info(f"Encoder attentions shape: {[att.shape for att in encoder_attentions]}")
        logger.info(f"Decoder attentions shape: {[att.shape for att in decoder_attentions]}")
        logger.info(f"Cross attentions shape: {[att.shape for att in cross_attentions]}")
    except Exception as e:
        logger.error("Error loading attention data: %s", str(e))
        return jsonify({"error": str(e)}), 500

    try:
        html_content = model_view(
            encoder_attention=encoder_attentions,
            decoder_attention=decoder_attentions,
            cross_attention=cross_attentions,
            encoder_tokens=encoder_text,
            decoder_tokens=generated_text_tokens,
            html_action='return'
        )
        logger.info("HTML content generated successfully")
        response = make_response(html_content.data)
        response.headers['Content-Type'] = 'text/html'
    except Exception as e:
        logger.error("Error generating model view: %s", str(e))
        return jsonify({"error": str(e)}), 500

    return response
