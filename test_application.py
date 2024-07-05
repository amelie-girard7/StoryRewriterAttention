import sys
import os
import pytest
import pandas as pd
import numpy as np  # Import numpy
from unittest import mock

# Add the directory containing application.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import application  # Import the main app file

@pytest.fixture
def client():
    application.app.config['TESTING'] = True
    with application.app.test_client() as client:
        yield client

def test_index(client):
    """Test the index route."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Story Visualization' in rv.data

def test_get_models(client):
    """Test the get_models route."""
    rv = client.get('/get_models')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert isinstance(json_data, list)
    assert json_data[0]['key'] == 'T5-base weight 1-1'

def test_get_stories(client, mocker):
    """Test the get_stories route."""
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['1']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/get_stories')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert isinstance(json_data, list)
    assert json_data[0]['Premise'] == 'Test Premise'

def test_fetch_story_data(client, mocker):
    """Test the fetch_story_data route."""
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['1']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/fetch_story_data', json={'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data['Premise'] == 'Test Premise'

def test_visualize_attention(client, mocker):
    """Test the visualize_attention route."""
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['1']
    })
    mock_attention_data = (
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # encoder_attentions
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # decoder_attentions
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # cross_attentions
        ['token1', 'token2'],  # encoder_text
        'generated text',  # generated_text
        ['gen_token1', 'gen_token2']  # generated_text_tokens
    )
    mocker.patch('application.load_data', return_value=mock_data)
    mocker.patch('application.get_attention_data', return_value=mock_attention_data)
    mocker.patch('application.plot_attention_heatmap', return_value=None)
    rv = client.post('/visualize_attention', json={'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data


def test_visualize_model_view(client, mocker):
    """Test the visualize_model_view route."""
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['1']
    })
    mock_attention_data = (
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # encoder_attentions
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # decoder_attentions
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # cross_attentions
        ['token1', 'token2'],  # encoder_text
        'generated text',  # generated_text
        ['gen_token1', 'gen_token2']  # generated_text_tokens
    )
    mocker.patch('application.load_data', return_value=mock_data)
    mocker.patch('application.get_attention_data', return_value=mock_attention_data)
    rv = client.post('/visualize_model_view', json={'story_index': 0, 'attention_type': 'cross'})
    assert rv.status_code == 200
    response_data = rv.data.decode('utf-8')
    print("Response data:", response_data)  # Debugging log
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>' in response_data
    assert '<div id="vis">' in response_data  # Adjust this based on actual analysis

if __name__ == '__main__':
    pytest.main()
