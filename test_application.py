import sys
import os
import pytest
import pandas as pd
import numpy as np
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
    assert json_data[0]['key'] == 'model_2024-03-22-10'

def test_visualize_attention(client, mocker):
    """Test the visualize_attention route."""
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mock_attention_data = (
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # encoder_attentions
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # decoder_attentions
        [np.random.rand(1, 12, 2, 2) for _ in range(12)],  # cross_attentions
        ['token1', 'token2'],  # encoder_text
        'generated text',  # generated_text
        ['gen_token1', 'gen_token2']  # generated_text_tokens
    )
    mocker.patch('heatmap_view.get_attention_data', return_value=mock_attention_data)
    mocker.patch('heatmap_view.plot_attention_heatmap', return_value=None)
    mocker.patch('application.load_data', return_value=mock_data)

    # Test for the first story ID in model_2024-03-22-10
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-03-22-10', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-03-22-10
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-03-22-10', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-03-22-15
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-03-22-15', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-03-22-15
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-03-22-15', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-04-08-09
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-08-09', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-04-08-09
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-08-09', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-04-08-13
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-08-13', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-04-08-13
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-08-13', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-04-09-11
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-09-11', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-04-09-11
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-09-11', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-04-09-22
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-09-22', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-04-09-22
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-09-22', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-04-10-10
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-10-10', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-04-10-10
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-10-10', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-04-10-14
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-10-14', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-04-10-14
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-04-10-14', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-05-13-17
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-05-13-17', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-05-13-17
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-05-13-17', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the first story ID in model_2024-05-14-20
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise'],
        'Initial': ['Test Initial'],
        'Original Ending': ['Test Original Ending'],
        'Counterfactual': ['Test Counterfactual'],
        'Edited Ending': ['Test Edited Ending'],
        'Generated Text': ['Test Generated Text'],
        'StoryID': ['9387e571-2819-4e29-bedb-a35f0410da51']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-05-14-20', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

    # Test for the second story ID in model_2024-05-14-20
    mock_data = pd.DataFrame({
        'Premise': ['Test Premise 2'],
        'Initial': ['Test Initial 2'],
        'Original Ending': ['Test Original Ending 2'],
        'Counterfactual': ['Test Counterfactual 2'],
        'Edited Ending': ['Test Edited Ending 2'],
        'Generated Text': ['Test Generated Text 2'],
        'StoryID': ['ca8a7f8d-7f63-422f-8007-c4a26bb8e889']
    })
    mocker.patch('application.load_data', return_value=mock_data)
    rv = client.post('/visualize_attention', json={'model_key': 'model_2024-05-14-20', 'story_index': 0})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'image_path' in json_data
    print("Image path:", json_data['image_path'])  # Debugging log

if __name__ == '__main__':
    pytest.main()
