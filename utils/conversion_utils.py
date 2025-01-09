from services.ScrapingBeeService import ScrapedContent
from PIL import Image
import base64
from typing import List, Dict, Any
import base64
from io import BytesIO
import fitz  # PyMuPDF

# TODO Refactor or make an adapter for this
def split_image(image_bytes: bytes, segment_height: int = 1500) -> list[bytes]:
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
                
    # Calculate number of segments needed

    num_segments = -(-height // segment_height)  # Ceiling division
    
    # Split into segments
    segments = []
    for i in range(num_segments):
        start_y = i * segment_height
        end_y = min(start_y + segment_height, height)
        
        # Crop the segment
        segment = img.crop((0, start_y, width, end_y))
        
        # Resize the segment to reduce resolution
        new_width = width // 3  # Reduce width by same factor as height (1500 -> 500)
        new_height = segment_height // 3
        segment = segment.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert segment to bytes
        segment_bytes = BytesIO()
        segment.save(segment_bytes, format='PNG')
        segments.append(segment_bytes.getvalue())
    
    return segments

def image_to_base64_string(image_bytes: bytes, format: str = "png") -> str:# Convert bytes to base64
    b64_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/{format};base64,{b64_data}"


def convert2message_parts(scraped_content: ScrapedContent) -> List[Dict[str, Any]]:
        """
        Converts ScrapedContent to OpenAI API compatible message parts.
        Returns a list because some content types might generate multiple parts.
        """
        if scraped_content.content_type.startswith("image/"):
            return convert_image(scraped_content)
        elif scraped_content.content_type == "application/pdf":
            return [convert_pdf(scraped_content)]
        elif scraped_content.content_type.startswith("text/"):
            return [convert_text(scraped_content)]
            
        raise ValueError(f"Unsupported content type: {scraped_content.content_type}")


def convert_text(content: ScrapedContent) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": content.content
    }


def convert_image(content: ScrapedContent) -> List[Dict[str, Any]]:
    if content.encoding == "utf-8":  # Handle base64 string input
        # Convert base64 string back to bytes
        image_bytes = base64.b64decode(content.content)
    else:  # Handle raw binary input
        image_bytes = content.content
        
    # Split the image into segments

    image_segments = split_image(image_bytes)
    
    # Convert each segment to a message part
    return [{
        "type": "image_url",
        "image_url": {
            "url": image_to_base64_string(segment)
        }
    } for segment in image_segments]


def convert_pdf(content: ScrapedContent) -> List[Dict[str, Any]]:
    # Create a BytesIO object from the content
    pdf_stream = BytesIO(content.content)
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    
    # Extract text from all pages
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    
    # Close the document
    doc.close()
    
    return [{
        "type": "text",
        "text": text
    }]