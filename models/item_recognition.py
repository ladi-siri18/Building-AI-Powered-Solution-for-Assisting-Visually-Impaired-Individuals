from PIL import Image
from collections import Counter
import cv2
import numpy as np

def recognize_items(image_path):
    """
    Recognizes and classifies items in the uploaded image.
    Condenses repetitive detections and provides more meaningful descriptions.
    """
    # Example output from your object detection model (this can vary based on your model)
    recognized_objects = ["person", "person", "handbag", "cell phone", "person", "person", "handbag", "cell phone", "backpack"]

    # Count the occurrences of each object
    object_counts = Counter(recognized_objects)

    # Provide more meaningful descriptions for each recognized item
    object_descriptions = {
        "person": "A person. This could be someone in the scene.",
        "handbag": "A handbag. It might contain essential items such as keys, phone, or wallet.",
        "cell phone": "A cell phone. Used for communication, browsing the web, etc.",
        "backpack": "A backpack. Often used to carry books, clothes, or other personal items."
    }

    # List the objects with descriptions and counts
    result = []
    for obj, count in object_counts.items():
        description = object_descriptions.get(obj, f"A {obj}.")
        result.append(f"{obj.capitalize()}: {description} (Found {count} times)")

    # Load the image to display with annotations
    image = Image.open(image_path)
    image_np = np.array(image)

    # Annotating the image (for example, using simple labels on the objects)
    for obj in object_counts.keys():
        # Example: Draw a simple label (you can add bounding boxes if required)
        cv2.putText(image_np, f'{obj.capitalize()}: {object_counts[obj]}',
                    (50, 50 + 30 * list(object_counts.keys()).index(obj)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save or process the annotated image
    annotated_image_path = "annotated_image.jpg"
    cv2.imwrite(annotated_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    return {
        "image": annotated_image_path,  # This would be the image path after annotation
        "items": result
    }
