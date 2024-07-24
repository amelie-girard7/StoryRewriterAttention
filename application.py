import os
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory, make_response
import pandas as pd

# Importing visualization functions from separate files
from heatmap_view import visualize_attention, plot_attention_heatmap, get_attention_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration classes
class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///:memory:')
    DATA_DIR = Path('data')  # Ensure this is a Path object

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True

# Initialize Flask app
app = Flask(__name__)

# Use environment variable to determine which configuration to use
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

# Use configuration values
DATA_DIR = Path(app.config['DATA_DIR'])

def load_data(file_path):
    """
    Load data from a CSV file.
    Returns a pandas DataFrame if the file exists, otherwise returns None.
    """
    if file_path is None or not file_path.exists():
        logger.error(f"Data path {file_path} does not exist.")
        return None
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

@app.route('/')
def index():
    """
    Render the main index page of the application.
    """
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Return a list of models available for selection.
    """
    logger.info("get_models endpoint called")
    # Updated model mappings
    model_mappings = {
        "model_2024-03-22-10": "T5-base weight 1-1",
        "model_2024-04-09-22": "T5-base weight 13-1",
        "model_2024-04-08-13": "T5-base weight 20-1",
        "model_2024-03-22-15": "T5-large weight 1-1",
        "model_2024-04-10-10": "T5-large weight 15-1",
        "model_2024-04-08-09": "T5-large weight 20-1",
        "model_2024-04-10-14": "T5-large weight 30-1",
        "model_2024-05-13-17": "T5-base weight 13-1 (Gold data)",
        "model_2024-05-14-20": "T5-base weight 20-1 (Gold data)"
    }
    models = [{"key": key, "comment": comment} for key, comment in model_mappings.items()]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    """
    Return a list of stories from the loaded data.
    """
    logger.info("get_stories endpoint called")
    model_key = request.json.get('model_key')
    if model_key is None:
        return jsonify({"error": "Model key not provided"}), 400

    data_path = DATA_DIR / model_key / 'test_data_sample-attention.csv'
    data = load_data(data_path)
    if data is None:
        logger.error("Data not found")
        return jsonify({"error": "Data not found"}), 404
    logger.info("Loaded data: %s", data.head())
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text', 'StoryID']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/fetch_story_data', methods=['POST'])
def fetch_story_data():
    """
    Return detailed information about a specific story given its index.
    """
    model_key = request.json.get('model_key')
    story_index = request.json.get('story_index')
    if model_key is None or story_index is None:
        return jsonify({"error": "Model key or Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data_path = DATA_DIR / model_key / 'test_data_sample-attention.csv'
    data = load_data(data_path)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    story = data.iloc[story_index].to_dict()
    return jsonify(story)

@app.route('/visualize_attention', methods=['POST'])
def visualize_attention_route():
    """
    Route to handle visualization of attention heatmap for a specific story.
    """
    logger.info("visualize_attention endpoint called")
    model_key = request.json.get('model_key')
    story_index = request.json.get('story_index')
    logger.info(f"Received model_key: {model_key}, story_index: {story_index}")  # Debug: Print received parameters
    
    if model_key is None or story_index is None:
        return jsonify({"error": "Model key or Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data_path = DATA_DIR / model_key / 'test_data_sample-attention.csv'
    data = load_data(data_path)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    story_id = data.iloc[story_index]["StoryID"]

    attention_path = DATA_DIR / model_key / 'attentions'
    image_path = generate_attention_image_path(model_key, story_id, DATA_DIR)

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

        plot_attention_heatmap(attention_to_plot, encoder_text, generated_text_tokens, "Cross-Attention Weights (First Layer)", image_path)
        logger.info(f"Generated heatmap image path: {image_path}")  # Debug: Print generated image path
    except Exception as e:
        logger.error("Error generating heatmap: %s", str(e))
        return jsonify({"error": str(e)}), 500

    return jsonify({"image_path": str(image_path)})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve the generated heatmap image from the centralized images directory.
    """
    images_dir = Path('images')
    return send_from_directory(images_dir, filename)

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
    images_dir = base_dir / 'images'
    return images_dir / f'{model_key}_{story_id}.png'

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
