import cv2
from image_preprocessing_cls import tensorflow_load_model_cls
import os
class TfModelSingleton:
    _instance = None

    def __new__(cls, model_path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_cls = tensorflow_load_model_cls(model_path=model_path)
        return cls._instance

class CascadeSingleton:
    _instance = None

    def __new__(cls, cascade_path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(),cascade_path))
        return cls._instance