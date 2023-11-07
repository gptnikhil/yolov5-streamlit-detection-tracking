import shutil
import os
import numpy as np
import cv2
import tensorflow as tf
from settings import CLASSES_NAME
def tensorflow_load_model_cls(model_path):
    """
    Loads a Keras model from a specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        tf.keras.Model: Loaded Keras model.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    """
    try:
        # Attempt to load the model
        model = tf.keras.models.load_model(os.path.join(os.getcwd(),model_path))
        return model
    except Exception as e:
        raise FileNotFoundError(f"Error loading model: {e}")

def preprocess_image(image):
    """
    Preprocesses an image for input into a MobileNetV2 model.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.

    Raises:
        ValueError: If the input image is not a valid NumPy array.

    """
    # Convert the input image to a TensorFlow tensor
    image = tf.convert_to_tensor(image)

    # Resize the image to (224, 224)
    image = tf.image.resize(image, (224, 224))

    # Preprocess the image using MobileNetV2 preprocessing function
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image


def read_image_as_pil(image_path):
    """
    Reads an image from a local file and returns it as a PIL (Pillow) object in RGB mode.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: A PIL Image object in RGB mode, or None if there was an error.
    """
    try:
        # Open the image and convert it to RGB mode
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error reading the image: {e}")
        return None



def detect_faces(img, face_cascade):
    """
    Detects faces in an image using a given face cascade.

    Args:
        img (numpy.ndarray): Input image in BGR format.
        face_cascade: Face cascade classifier for face detection.

    Returns:
        list: List of detected face images.
    """
    img_rgb = np.array(img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Detect faces using the provided face cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print(faces)

    list_faces = []

    # Loop through detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face from the original image
        face = img_rgb[y:y+h, x:x+w]

        # Apply Gaussian blur to the face
        face = cv2.GaussianBlur(face, (0, 0), sigmaX=1)

        # Add the processed face to the list
        list_faces.append(face)
   


    return list_faces


def predict_image(image, model):
    """
    Preprocesses an image and makes a prediction using a specified model.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        model (tf.keras.Model): The model used for making predictions.

    Returns:
        dict: Predicted Result 

    Raises:
        ValueError: If the input image is not a valid NumPy array.

    """
    # Preprocess the image
    image = preprocess_image(image)

    # Make a prediction using the specified model
    predictions = model.predict(tf.expand_dims(image, axis=0))
    print(type(predictions))

    predicted_values = predictions[0]  # Assuming predictions is a 2D array

    result_dict = {class_name: [predicted_value] for class_name, predicted_value in zip(CLASSES_NAME, predicted_values)}

    return result_dict


def predict_all_faces(list_faces, model):
    """
    Preprocesses a list of faces and makes predictions for each face using a specified model.

    Args:
        list_faces (list of numpy.ndarray): List of input images as NumPy arrays.
        model (tf.keras.Model): The model used for making predictions.

    Returns:
        list of numpy.ndarray: List of predicted class probabilities for each face.

    Raises:
        ValueError: If any of the input images are not valid NumPy arrays.

    """
    all_predictions = []
    values=[]
    try:

        for image in list_faces:
            # Preprocess the image
            image = preprocess_image(image)

            # Make a prediction using the specified model
            predictions = model.predict(tf.expand_dims(image, axis=0))
            # print(predictions)
            
            index_class=np.argmax(predictions)

            all_predictions.append(index_class)
            values.append(predictions[index_class])

        final_class_index=max(all_predictions)

        result={CLASSES_NAME[final_class_index]:[np.max(values)]}
     

        return result
    except Exception as e:
        print(f"No Face Detection{e}")
        return None




def load_face_cascade(cascade_path):
    """
    Loads a face cascade classifier from a specified file.

    Args:
        cascade_path (str): Path to the XML file containing the face cascade.

    Returns:
        cv2.CascadeClassifier: Loaded face cascade classifier.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    """
    try:
        # Attempt to load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2cascade_path)
        
        return face_cascade
    except Exception as e:
        raise FileNotFoundError(f"Error loading face cascade classifier: {e}")

# # Example usage:
# cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = load_face_cascade(cascade_path)
