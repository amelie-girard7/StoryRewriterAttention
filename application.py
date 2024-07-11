import os
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory, make_response
import pandas as pd
import numpy as np

# Importing visualization functions from separate files
from heatmap_view import visualize_attention, plot_attention_heatmap
from model_view import visualize_model_view

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration classes
class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///:memory:')
    DATA_DIR = 'data'  # Base directory for models

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
    model_mappings = {
        "model_2024-03-22-10": "T5-base weight 1-1",
        "model_2024-04-09-22": "T5-base weight 13-1",
        "model_2024-04-08-13": "T5-base weight 20-1",
        "model_2024-05-13-17": "T5-base weight 13-1 (Gold data)"
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
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text']].to_dict(orient='records')
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
    return visualize_attention(request, DATA_DIR, load_data, logger, plot_attention_heatmap)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve the generated heatmap image.
    """
    return send_from_directory('/tmp', filename)

@app.route('/visualize_model_view', methods=['POST'])
def visualize_model_view_route():
    """
    Route to handle visualization of model view for the attention mechanism.
    """
    return visualize_model_view(request, DATA_DIR, load_data, logger)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
