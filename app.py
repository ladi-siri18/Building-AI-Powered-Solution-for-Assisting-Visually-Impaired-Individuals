import streamlit as st
import time  # Added to resolve time.sleep() issue
from PIL import Image
import pytesseract
from ultralytics import YOLO
from pytesseract import pytesseract
from transformers import DetrImageProcessor, DetrForObjectDetection  # Added for DETR model
import torch
import numpy as np
import cv2
from models.file_handler import save_uploaded_file
from models.scene import generate_caption
from models.ocr import extract_text, text_to_speech

# Specify the Tesseract executable path
pytesseract.tesseract_cmd = r"C:\Users\ladis\final\Tesseract-OCR\tesseract.exe"  # Update for your system

# Hugging Face API Configuration
HF_API_KEY = "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"  # Your Hugging Face API Key
HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# YOLO Model for Object Detection
yolo_model = YOLO("yolov8n.pt")  # Lightweight version of YOLOv8

# Initialize the DETR Model for object detection
model_name = "facebook/detr-resnet-50"  # You can use other models from Hugging Face as well
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Function for Object Detection using DETR model
def detect_objects(image_path):
    """
    Detect objects in an image using the DETR model and draw bounding boxes on detected objects.
    """
    try:
        # Open the image and ensure it is in RGB format
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image for the model
        inputs = processor(images=image, return_tensors="pt")

        # Perform object detection
        with torch.no_grad():  # Disabling gradients as it's not required for inference
            outputs = model(**inputs)

        # Get results: The model outputs bounding boxes and labels
        target_sizes = torch.tensor([image.size[::-1]])  # Convert image size (height, width)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5  # Lowered threshold to 0.5 for better detection
        )[0]

        # Check if results are available
        if len(results["scores"]) == 0:
            return {"error": "No objects detected."}

        # Convert the image to a numpy array and ensure compatibility with OpenCV (convert to BGR)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create an image with bounding boxes around the detected objects
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]  # Extract box coordinates
            # Draw the bounding box
            cv2.rectangle(
                image_np,
                (int(box[0]), int(box[1])),  # Top-left corner
                (int(box[2]), int(box[3])),  # Bottom-right corner
                (255, 0, 0),  # Color: Blue
                3,  # Thickness
            )
            # Add the label and score
            cv2.putText(
                image_np,
                f'{model.config.id2label[label.item()]}: {round(score.item(), 3)}',
                (int(box[0]), int(box[1]) - 10),  # Position above the bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,  # Font scale
                (255, 255, 255),  # Font color: White
                2,  # Thickness
            )

        # Save the image with bounding boxes
        output_image_path = "output_image_with_bboxes.jpg"
        cv2.imwrite(output_image_path, image_np)

        # Extract object labels and scores for the user
        object_labels = [
            f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            for score, label in zip(results["scores"], results["labels"])
        ]

        return {"image": output_image_path, "objects": object_labels}

    except Exception as e:
        return {"error": str(e)}

# Streamlit App
st.title("AI-Powered Assistance for Visually Impaired Individuals")
st.subheader("Upload an image to get assistive features!")

# Upload Image Section (using one uploader only)
uploaded_image = st.file_uploader("Upload an Image 📷", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Save image to a temporary location
    file_path = save_uploaded_file(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Choose functionality
    option = st.radio("Select a functionality:", 
                      ("Real-Time Scene Understanding", 
                       "Text-to-Speech Conversion",
                       "Object-detection",
                       ))

    if option == "Real-Time Scene Understanding":
        # Display a progress bar while generating the caption
        st.write("Generating Scene Description... Please wait.")
        progress = st.progress(0)  # Initialize the progress bar

        # Simulate processing time with a loop (this is just for demonstration)
        for i in range(100):
            time.sleep(0.05)  # Simulate some work being done (e.g., model inference)
            progress.progress(i + 1)  # Update the progress bar

        # Once progress bar is done, generate the caption
        caption = generate_caption(file_path)
        st.write("Scene Description 🖼️:")
        st.write(caption)
    
    elif option == "Text-to-Speech Conversion":
        # Display a progress bar while extracting text and converting to speech
        st.write("Extracting Text... Please wait.")
        progress = st.progress(0)

        # Simulate processing time with a loop
        for i in range(100):
            time.sleep(0.05)  # Simulate some work (OCR or text-to-speech)
            progress.progress(i + 1)  # Update the progress bar

        # Once progress bar is done, extract text and convert to speech
        text = extract_text(file_path)
        if text:
            st.write("Extracted Text 📝:")
            st.write(text)
            st.write("Playing the text as speech 🎤...")
            text_to_speech(text)
        else:
            st.write("No text found in the image.")

    elif option == "Object-detection":
        # Display a progress bar while detecting objects
        st.write("Detecting Objects... Please wait.")
        progress = st.progress(0)

        # Simulate processing time with a loop (adjust with actual processing time)
        for i in range(100):
            time.sleep(0.05)  # Simulate some work
            progress.progress(i + 1)

        # Once progress bar is done, detect objects in the uploaded image
        detected_objects = detect_objects(file_path)

        # Show the image with detected objects
        st.write("Objects Detected 🔍:")
        if "image" in detected_objects:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)


        # Display detected object names and descriptions
        st.write("Detected Objects List 🏷️:")
        if "objects" in detected_objects and detected_objects["objects"]:
            for obj in detected_objects["objects"]:
                st.write(f"- {obj}")
                text_to_speech(obj)
        else:
            st.write("No objects detected in the image.")

    else:
        st.write("No functionality selected.")
