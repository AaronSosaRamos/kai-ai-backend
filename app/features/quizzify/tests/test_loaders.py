
import os
import pytest
from unittest.mock import patch, MagicMock
from app.features.quizzify.document_loaders import load_docs_youtube_url

# Mocking necessary components
@patch('app.features.quizzify.document_loaders.generate_image_summaries')  # Mocking generate_image_summaries
@patch('app.features.quizzify.document_loaders.extract_image_frames')  # Mocking extract_image_frames
@patch('app.features.quizzify.document_loaders.YouTube')
@patch('app.features.quizzify.document_loaders.YoutubeLoader')
@patch('app.features.quizzify.document_loaders.splitter')
@patch('app.features.quizzify.document_loaders.shutil.rmtree')
def test_load_docs_youtube_url(mock_shutil_rmtree, mock_splitter, mock_YoutubeLoader, mock_YouTube, mock_extract_image_frames, mock_generate_image_summaries):
    
    # Mocking YouTube video download
    mock_youtube = MagicMock()
    mock_youtube.streams.get_highest_resolution.return_value.download.return_value = 'mock_video.mp4'
    mock_YouTube.return_value = mock_youtube
    
    # Mocking YoutubeLoader behavior
    mock_loader = MagicMock()
    mock_loader.load.return_value = [{'page_content': 'mock_audio_content', 'metadata': {}}]  # Return a dictionary
    mock_YoutubeLoader.from_youtube_url.return_value = mock_loader

    # Mocking document splitting for audio
    mock_splitter.split_documents.return_value = [{'page_content': 'mock_audio_content', 'metadata': {}}]  # Return a dictionary

    # Mocking generate_image_summaries to return dummy image document objects
    mock_generate_image_summaries.return_value = [
        {'page_content': "Image summary 1", 'metadata': {"image_url": "mock_image1.jpg"}},
        {'page_content': "Image summary 2", 'metadata': {"image_url": "mock_image2.jpg"}}
    ]

    # Call the function
    result = load_docs_youtube_url('mock_youtube_url')

    # Assertions
    assert len(result) == 3  # Assuming 2 image docs + 1 audio doc
    assert result[0]['page_content'] == "Image summary 1"
    assert result[1]['page_content'] == "Image summary 2"
    assert result[2]['page_content'] == "mock_audio_content"  # Check the dict key for audio content

    # Ensure generate_image_summaries was called with the correct path
    mock_generate_image_summaries.assert_called_once_with(os.path.join((os.path.dirname(os.path.abspath("./app"))), "app/features/quizzify/video_data/video_img"))

    # Additional assertions to verify that mocks were called correctly
    mock_youtube.streams.get_highest_resolution.assert_called_once()
    mock_youtube.streams.get_highest_resolution.return_value.download.assert_called_once()
    mock_loader.load.assert_called_once()
    mock_splitter.split_documents.assert_called_once()
    mock_shutil_rmtree.assert_called_once_with('./app/features/quizzify/video_data')
