import os

def save_uploaded_file(uploaded_file):
    # Define the temp directory where you want to store the file
    temp_dir = "temp"
    
    # Check if the temp directory exists, and create it if it doesn't
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the 'temp' directory
    
    # Generate a valid file name by sanitizing the uploaded file name (in case of special characters)
    file_name = uploaded_file.name
    file_path = os.path.join(temp_dir, file_name)
    
    # Check if file already exists to avoid overwriting
    if os.path.exists(file_path):
        # If the file already exists, add a unique identifier to the filename
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_path):
            file_name = f"{base}_{counter}{ext}"
            file_path = os.path.join(temp_dir, file_name)
            counter += 1

    # Write the file to the temp directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save the uploaded file
    
    return file_path  # Return the path of the saved file
