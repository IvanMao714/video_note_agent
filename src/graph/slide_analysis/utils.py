import asyncio
import base64
import io
from typing import Dict, List, Any

from langchain.agents import AgentState
from pdf2image import convert_from_path
from PIL import Image


from llms.llm import get_llm_by_type
from log import get_logger
from prompts.template import apply_prompt_template, get_prompt_template

logger = get_logger(__name__)


def encode_image(image_obj: Image.Image) -> str:
    """
    Encode a PIL Image object to a base64-encoded JPEG string.

    The image is converted to JPEG format and then encoded as a base64 string,
    suitable for embedding in multimodal LLM messages (e.g., OpenAI's image_url format).

    Args:
        image_obj: PIL Image object to encode. Must be a valid PIL Image instance.

    Returns:
        Base64-encoded string representation of the JPEG image, ready for use
        in data URI format (e.g., "data:image/jpeg;base64,{base64_string}").

    Raises:
        IOError: If image encoding or conversion to JPEG fails.
    """
    try:
        buffered = io.BytesIO()
        image_obj.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    except Exception as e:
        raise IOError(f"Failed to encode image: {str(e)}") from e


async def analyze_single_page(page_index: int, image: Image.Image, total_pages: int) -> Dict[str, Any]:
    """
    Analyze a single slide page using a multimodal LLM.

    This function encodes the page image to base64, constructs a multimodal message
    with the slide image and analysis prompt, and invokes a vision-capable LLM to
    generate structured analysis of the slide content.

    Args:
        page_index: Zero-based index of the page being analyzed.
        image: PIL Image object representing the page image to analyze.
        total_pages: Total number of pages in the PDF presentation.

    Returns:
        Dictionary containing:
            - page_number (int): One-based page number.
            - content (str): Analysis result from the LLM, or error message if analysis failed.

    Note:
        This function requires a vision-capable LLM configured in the system.
        The LLM is invoked asynchronously to support parallel processing of multiple pages.
    """

    # Encode image to base64
    try:
        img_base64 = encode_image(image)
        if not img_base64 or len(img_base64) == 0:
            error_msg = f"Empty base64 encoding for page {page_index + 1}"
            logger.error(error_msg)
            return {
                "page_number": page_index + 1,
                "content": f"Error: {error_msg}"
            }
        logger.debug(f"Encoded image for page {page_index + 1}, base64 length: {len(img_base64)}")
    except Exception as e:
        error_msg = f"Failed to encode image for page {page_index + 1}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "page_number": page_index + 1,
            "content": f"Error: {error_msg}"
        }

    state = AgentState(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze this slide (Page {page_index + 1} of {total_pages})."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    }
                ]
            }
        ]
    )
    
    try:
        messages = apply_prompt_template("slide_analyzer", state)
        logger.debug(f"Applied prompt template for page {page_index + 1}, got {len(messages)} messages")
    except Exception as e:
        error_msg = f"Failed to apply prompt template for page {page_index + 1}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "page_number": page_index + 1,
            "content": f"Error: {error_msg}"
        }

    # Invoke LLM - requires vision-capable LLM to support image input
    # get_llm_by_type may trigger blocking calls, move to thread
    llm = await asyncio.to_thread(get_llm_by_type, "vision")
    logger.info(f"Using vision LLM for page {page_index + 1}/{total_pages}")

    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        error_msg = f"LLM invocation failed for page {page_index + 1}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "page_number": page_index + 1,
            "content": f"Error: {error_msg}"
        }

    return {
        "page_number": page_index + 1,
        "content": response.content
    }


def get_slides_images(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract images from PDF slides and encode them to base64.
    
    This function converts each PDF page to an image and encodes them as base64 strings
    for use in multimodal LLM messages.
    
    Args:
        pdf_path: Path to the input PDF file.
        
    Returns:
        List of dictionaries, each containing:
            - page_number (int): One-based page number.
            - image_base64 (str): Base64-encoded JPEG image string.
        Results are sorted by page_number in ascending order.
        
    Raises:
        RuntimeError: If PDF conversion to images fails.
    """
    logger.info(f"Extracting images from PDF: {pdf_path}")
    
    try:
        images = convert_from_path(pdf_path)
        total_pages = len(images)
        logger.info(f"Loaded {total_pages} slides for image extraction.")
    except Exception as e:
        error_msg = f"Failed to convert PDF to images: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    
    slides_images = []
    for i, img in enumerate(images):
        try:
            img_base64 = encode_image(img)
            slides_images.append({
                "page_number": i + 1,
                "image_base64": img_base64
            })
        except Exception as e:
            logger.error(f"Failed to encode image for page {i + 1}: {e}", exc_info=True)
            # Continue with other pages even if one fails
            continue
    
    logger.info(f"Successfully extracted {len(slides_images)} slide images")
    return slides_images


async def analyze_presentation(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Analyze an entire presentation PDF by processing all pages in parallel.

    This function converts each PDF page to an image, then analyzes all pages
    concurrently using asyncio.gather for improved performance. Results are
    sorted by page number to maintain order.

    Args:
        pdf_path: Path to the input PDF file to analyze.

    Returns:
        List of dictionaries, each containing:
            - page_number (int): One-based page number.
            - content (str): Analysis result for that page, or error message if analysis failed.
        Results are sorted by page_number in ascending order.

    Raises:
        RuntimeError: If PDF conversion to images fails.

    Note:
        All pages are analyzed in parallel for maximum throughput. The function
        uses asyncio.gather to coordinate concurrent LLM invocations.
    """
    logger.info(f"Analyzing presentation from: {pdf_path} by analyze_presentation function")

    try:
        images = convert_from_path(pdf_path)
        total_pages = len(images)
        logger.info(f"Loaded {total_pages} slides.")
    except Exception as e:
        error_msg = f"Failed to convert PDF to images: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # Step B: Create task list
    tasks = []
    for i, img in enumerate(images):
        tasks.append(analyze_single_page(i, img, total_pages))

    # Step C: Execute in parallel (Map phase)
    # asyncio.gather allows all pages to be analyzed concurrently for maximum speed
    results = await asyncio.gather(*tasks)

    # Step D: Sort results by page number (ensures order despite concurrent execution)
    results.sort(key=lambda x: x['page_number'])
    # results = []

    return results

if __name__ == '__main__':
    pdf_file_path = "E:\\OpencourseAgent\\example\\02-map-reduce.pdf"  # Replace with your PDF file path
    analysis_results = asyncio.run(analyze_presentation(pdf_file_path))
    for page_analysis in analysis_results:
        logger.info(f"Page {page_analysis['page_number']} Analysis:\n{page_analysis['content']}\n")