from app.features.dynamo.tools import summarize_transcript, generate_flashcards
from app.services.logger import setup_logger
from app.api.error_utilities import VideoTranscriptError

logger = setup_logger(__name__)

def executor(youtube_url: str, verbose=False):
    summary, title = summarize_transcript(youtube_url, verbose=verbose)
    flashcards = generate_flashcards(summary)

    sanitized_flashcards = []
    for flashcard in flashcards:
        if 'concept' in flashcard and 'definition' in flashcard:
            sanitized_flashcards.append({
                "concept": flashcard['concept'],
                "definition": flashcard['definition']
            })
        else:
            logger.warning(f"Malformed flashcard skipped: {flashcard}")

    final_result = {
        "details": {
            "title": title,
            "description": summary,
        },
        "concepts": sanitized_flashcards
    } 

    return final_result