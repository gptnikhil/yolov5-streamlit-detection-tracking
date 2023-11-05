import streamlit as st
import cv2
import numpy as np
import shutil
import os
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
from settings import CLASSES_NAME
import  tempfile
from settings import  CLASSES_NAME


class ObjectProcessor:
    def __init__(self):
        self.temp_folder = os.path.join(tempfile.gettempdir(), "detected_objects")
        
        if os.path.exists(self.temp_folder):
    # If it exists, remove it and its contents
            shutil.rmtree(self.temp_folder)

        os.makedirs(self.temp_folder)
        for class_idx in CLASSES_NAME:
            class_temp_folder = os.path.join(self.temp_folder, f'class_{class_idx}')
            os.makedirs(class_temp_folder, exist_ok=True)
        self.counter=0


    def process_frame(self, frame,results):
        
        for idx, result in results.iterrows():
            print(result.to_dict())
            x1, y1, x2, y2, class_pred = result['xmin'],result['ymin'],result['xmax'],result["ymax"],result['name']
         
            detected_object = frame[int(y1):int(y2), int(x1):int(x2)]
            print("saved image")

            cv2.imwrite(os.path.join(self.temp_folder, f'class_{class_pred}',f'detected_object_{class_pred}_{self.counter}.jpg'), detected_object)
            
            self.counter+=1
            

    def download_detection_folder_zip(self):
     

        zip_file_name = "class_folder"
        shutil.make_archive(zip_file_name, 'zip', self.temp_folder)
        
        # Move the zip file to a publicly accessible directory
        target_zip_file_path = os.path.join(os.getcwd(), f"{zip_file_name}.zip")
        os.rename(f"{zip_file_name}.zip", target_zip_file_path)

        # Provide a download link to the zip file
       
        with open(f"{zip_file_name}.zip", "rb") as fp:
            btn = st.download_button(
                label="Download Class Cropped Objects",
                data=fp,
                file_name=f"{zip_file_name}.zip",
                mime="application/zip"
            )
        # Clean up temporary folder
        shutil.rmtree(self.temp_folder)
        os.remove(target_zip_file_path)

# Usage example:




