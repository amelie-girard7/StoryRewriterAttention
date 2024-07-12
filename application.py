import os
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_file
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
    DATA_DIR = '/data/agirard/Projects/StoryRewriterAttention/data'

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

# Hardcoded image paths dictionary for the specified model
IMAGE_PATHS = {
    "model_2024-03-22-10": {
        "0": "/data/agirard/Projects/StoryRewriterAttention/data/model_2024-03-22-10/attentions/ca8a7f8d-7f63-422f-8007-c4a26bb8e889/attention_heatmap_ca8a7f8d-7f63-422f-8007-c4a26bb8e889.png",
        "1": "/data/agirard/Projects/StoryRewriterAttention/data/model_2024-03-22-10/attentions/9387e571-2819-4e29-bedb-a35f0410da51/attention_heatmap_9387e571-2819-4e29-bedb-a35f0410da51.png"
    }
}

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
    model_mappings = {
        "model_2024-03-22-10": "T5-base weight 1-1"
    }
    models = [{"key": key, "comment": comment} for key, comment in model_mappings.items()]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    """
    Return a list of stories available for the given model.
    """
    logger.info("get_stories endpoint called")
    model_key = request.json.get('model_key')
    if model_key is None or model_key != 'model_2024-03-22-10':
        return jsonify({"error": "Model key not provided or invalid"}), 400

    data_path = DATA_DIR / model_key / 'test_data_sample-attention.csv'
    data = load_data(data_path)
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    logger.info("Loaded data: %s", data.head())
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/fetch_story_data', methods=['POST'])
def fetch_story_data():
    """
    Return detailed information about a specific story given its index.
    """
    model_key = request.json.get('model_key')
    story_index = request.json.get('story_index')
    if model_key is None or story_index is None or model_key != 'model_2024-03-22-10':
        return jsonify({"error": "Model key or Story index not provided or invalid"}), 400

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
    if model_key is None or story_index is None or model_key != 'model_2024-03-22-10':
        return jsonify({"error": "Model key or Story index not provided or invalid"}), 400
    
    response = visualize_attention(request, DATA_DIR, load_data, logger, plot_attention_heatmap)
    
    if response.status_code == 200:
        story_id = str(story_index)  # Convert to string to match the dictionary keys
        IMAGE_PATHS[model_key][story_id] = response.json['image_path']  # Update dictionary with the new path
    
    return response

@app.route('/images/<model_key>/<story_id>')
def serve_image(model_key, story_id):
    """
    Serve the generated heatmap image from the model-specific directory.
    """
    logger.info(f"Requesting image for model_key: {model_key}, story_id: {story_id}")
    if model_key != 'model_2024-03-22-10':
        return jsonify({"error": "Model key not provided or invalid"}), 400

    image_path = IMAGE_PATHS.get(model_key, {}).get(story_id)
    logger.info(f"Fetched image path from dictionary: {image_path}")
    
    if image_path is None:
        logger.error(f"Image path is None for model_key: {model_key}, story_id: {story_id}")
        return jsonify({"error": "Image not found"}), 404
    
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist on the filesystem: {image_path}")
        return jsonify({"error": "Image not found"}), 404

    logger.info(f"Serving image from path: {image_path}")
    return send_file(image_path)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
