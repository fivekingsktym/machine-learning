from keras.applications.vgg16 import VGG16 # VGG16 is a smart model that can recognize objects in images.
from keras.preprocessing import image # The image tool helps us load and work with pictures.
from keras.applications.vgg16 import preprocess_input, decode_predictions # preprocess_input and decode_predictions help prepare the picture and understand the robot's answers.
import numpy as np # NumPy is like a calculator that helps with any math we need to do.


# Load the pre-trained VGG16 model with weights trained on ImageNet dataset
model = VGG16(weights='imagenet')

def detect_cat(image_path):
    """
    Function to detect if a cat is present in the given image.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    (bool, float): Tuple containing a boolean indicating if a cat is detected
                   and the confidence score if a cat is detected, otherwise None.
    """
    
    # Load the image from the specified path, resizing it to 224x224 pixels (the input size expected by VGG16)
    img = image.load_img(image_path, target_size=(224, 224))
    
    # Convert the image to a NumPy array with shape (224, 224, 3) (3 channels for RGB)
    x = image.img_to_array(img)
    
    # Expand the dimensions of the array to (1, 224, 224, 3) to match the input shape expected by the model
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image array to match the format required by VGG16 (e.g., scaling pixel values)
    x = preprocess_input(x)

    # Make predictions on the preprocessed image using the VGG16 model
    preds = model.predict(x)
    
    # Decode the top 3 predictions into human-readable labels and their corresponding confidence scores
    decoded_preds = decode_predictions(preds, top=10)[0]
    
    # Iterate over the top 3 predictions
    for _, label, score in decoded_preds:
        # Check if any of the predicted labels contain the word 'cat'
        if 'cat' in label:
            # If a cat is detected, return True and the confidence score
            return True, score
    
    # If no cat is detected in the top 3 predictions, return False and None
    return False, None

# Test the function
import os

for item in os.listdir('/workspace/machine-learning/cat_detection/images/'):
    print("\n",item)

    cat_detected, confidence = detect_cat(f'/workspace/machine-learning/cat_detection/images/{item}')
    if cat_detected:
        print("Cat found with confidence:", confidence)
    else:
        print("No cat found in the image.")
