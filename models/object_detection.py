import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import cv2

# Load a pre-trained object detection model from Hugging Face
model_name = "facebook/detr-resnet-50"  # You can use other models from Hugging Face as well
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Function for object detection
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

# Example usage
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    result = detect_objects(image_path)
    print(result)
