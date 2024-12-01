import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import pyttsx3

# Set the correct Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ladis\final\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

def extract_text(image_path):
    """
    Extracts text from an image using OCR (Tesseract) with preprocessing.
    """
    try:
        # Check if the image file exists and is valid
        image = Image.open(image_path)
        
        # Preprocess the image: Convert to grayscale, sharpen, and enhance contrast
        image = image.convert("L")  # Convert to grayscale
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Enhance contrast
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(image)
        text = text.strip()  # Remove unnecessary whitespace
        
        # Validate the extracted text
        if not text or text.isspace():
            return "No readable text found in the image."
        
        return text
    except FileNotFoundError:
        return "Error: The specified image file does not exist."
    except pytesseract.TesseractError as e:
        return f"Error in Tesseract OCR: {e}"  # Specific error for Tesseract issues
    except Exception as e:
        return f"Error in extracting text: {e}"


def text_to_speech(text):
    """
    Convert the given text to speech using pyttsx3.
    """
    try:
        # Initialize the pyttsx3 engine
        engine = pyttsx3.init()
        
        # Configure voice properties
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Select the first voice (typically male)
        engine.setProperty('rate', 150)  # Set speech rate (words per minute)
        engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)
        
        # Convert the text to speech
        engine.say(text)
        engine.runAndWait()
    except RuntimeError as e:
        return f"Error: Failed to initialize pyttsx3 engine. {e}"  # Specific error for engine issues
    except Exception as e:
        return f"Error in text-to-speech: {e}"


# Example Usage
if __name__ == "__main__":
    # Replace with the path to your image
    image_path = "test_image.jpg"
    
    # Extract text from the image
    extracted_text = extract_text(image_path)
    print("Extracted Text:", extracted_text)
    
    # Convert text to speech
    if extracted_text and extracted_text != "No readable text found in the image.":
        text_to_speech(extracted_text)
    else:
        print("No text to convert to speech.")
