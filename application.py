from flask import Flask, jsonify, request, render_template, send_from_directory
import os
import pandas as pd
from pathlib import Path

# Configuration class
class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///:memory:')
    DATA_DIR = Path('/data/agirard/Projects/StoryRewriterAttention/data')
    IMAGE_DIR = Path('/data/agirard/Projects/StoryRewriterAttention/images')

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Use configuration values
app.config.from_object(Config)
DATA_DIR = app.config['DATA_DIR']
IMAGE_DIR = app.config['IMAGE_DIR']

def load_data(file_path):
    if file_path is None or not file_path.exists():
        return None
    return pd.read_csv(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    model_mappings = {
        "model_2024-03-22-10": "T5-base weight 1-1",
        "model_2024-04-09-22": "T5-base weight 13-1",
        "model_2024-04-08-13": "T5-base weight 20-1",
        "model_2024-03-22-15": "T5-large weight 1-1",
        "model_2024-04-10-10": "T5-large weight 15-1",
        "model_2024-04-08-09": "T5-large weight 20-1",
        "model_2024-04-10-14": "T5-large weight 30-1"
        #"model_2024-05-13-17": "T5-base weight 13-1 (Gold data)"
    }
    models = [{"key": key, "comment": comment} for key, comment in model_mappings.items()]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    model_key = request.json.get('model_key')
    if model_key is None:
        return jsonify({"error": "Model key not provided"}), 400

    data_path = DATA_DIR / model_key / 'test_data_sample-attention.csv'
    data = load_data(data_path)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text', 'StoryID']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/fetch_story_data', methods=['POST'])
def fetch_story_data():
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

    story_id = data.iloc[story_index]["StoryID"]
    image_path = IMAGE_DIR / f"{model_key}_{story_id}.png"

    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404

    return jsonify({"image_path": f"/static/images/{model_key}_{story_id}.png"})

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
