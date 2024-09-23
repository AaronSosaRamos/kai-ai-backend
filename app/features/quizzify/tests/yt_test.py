import pytest
from unittest.mock import patch, MagicMock
from app.features.quizzify.document_loaders import load_docs_youtube_url

# Mocking necessary components
@patch('app.features.quizzify.document_loaders.YouTube')
@patch('app.features.quizzify.document_loaders.YoutubeLoader')
@patch('app.features.quizzify.document_loaders.splitter')
@patch('app.features.quizzify.document_loaders.shutil.rmtree')
def test_load_docs_youtube_url(mock_shutil_rmtree, mock_splitter, mock_YoutubeLoader, mock_YouTube):
    # Mock objects
    mock_youtube = MagicMock()
    mock_youtube.streams.get_highest_resolution.return_value.download.return_value = 'mock_video.mp4'
    mock_youtube.return_value = mock_youtube
    
    mock_loader = MagicMock()
    mock_loader.load.return_value = [{'page_content': 'mock_content', 'metadata': {'image_url': 'mock_image.jpg'}}]
    mock_YoutubeLoader.from_youtube_url.return_value = mock_loader

    mock_splitter.split_documents.return_value = [{'page_content': 'mock_audio_content', 'metadata': {}}]

    # Call the function
    result = load_docs_youtube_url('mock_youtube_url')

    # Assertions
    assert len(result) == 2  # Assuming it returns a list of image_docs and audio_docs
    assert result[0]['page_content'] == 'mock_content'
    assert 'video_data' in result[0]['metadata']['image_url']
    assert result[1]['page_content'] == 'mock_audio_content'

    # Additional assertions for function behavior
    mock_youtube.streams.get_highest_resolution.assert_called_once()
    mock_youtube.streams.get_highest_resolution.return_value.download.assert_called_once()
    mock_loader.load.assert_called_once()
    mock_splitter.split_documents.assert_called_once()
    mock_shutil_rmtree.assert_called_once_with('./app/features/quizzify/video_data')

