import requests

from log import get_logger

logger = get_logger(__name__)


def get_text_from_bailian_result(result: dict, cache=True) -> str:
    """Extract text content from Bailian ASR transcription result.

    This function processes the result dictionary from Bailian ASR API, downloads
    the detailed transcription from the provided URL, and extracts the plain text
    content from the JSON response. The transcription results are typically stored
    in the 'transcripts' list within the downloaded data.

    Args:
        result: Dictionary containing ASR transcription result. Expected to have
            a 'results' key containing a list with at least one item that has a
            'transcription_url' key.

    Returns:
        str: Extracted plain text content from the transcription. Each transcript
            item's text is concatenated with newlines. Returns an error message
            string if the download or parsing fails, or if no results are found.

    Note:
        The function handles HTTP errors and JSON parsing exceptions, returning
        error messages as strings rather than raising exceptions.
    """
    # Extract download URL from result
    if not result.get('results'):
        return "No result link found"

    download_url = result['results'][0]['transcription_url']
    logger.info(f"Downloading detailed results: {download_url}")

    try:
        # Download content via HTTP GET request
        response = requests.get(download_url)
        response.raise_for_status()  # Check if request was successful

        # Parse the downloaded JSON
        data = response.json()


        # Extract plain text
        # Alibaba DashScope results are typically in the 'transcripts' list
        full_text = ""
        if 'transcripts' in data:
            for item in data['transcripts']:
                # Each item typically contains text, sentences, timestamps, etc.
                full_text += item.get('text', '') + "\n"

        return full_text

    except Exception as e:
        return f"Download or parsing failed: {str(e)}"