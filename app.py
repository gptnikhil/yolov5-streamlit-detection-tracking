# Python In-built packages
from pathlib import Path
import PIL
import tensorflow as tf
# External packages
import streamlit as st
import pandas as pd
from model_singleton import  TfModelSingleton,CascadeSingleton

# Local Modules
import settings
import helper
from image_preprocessing_cls import *
# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv5s",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.sidebar.header("ML Model Config")

classification=False
# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Classification'])

# confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 25, 100, 40)) / 100
# model = helper.custom_load_model(Path(settings.DETECTION_MODEL))
# model_cls = tensorflow_load_model_cls(model_path= Path(settings.CLASSIFICATION_MODEL))
# Selecting Detection Or Segmentation
if model_type == 'Detection':
    st.title("Object Detection using YOLOv5s")

# Sidebar
 
    model_path = Path(settings.DETECTION_MODEL)
    # Load Pre-trained ML Model
    try:
        model = helper.custom_load_model(model_path)
        print("Detection")
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
elif model_type == 'Classification':
    model_path_classification = Path(settings.CLASSIFICATION_MODEL)
    try:
        classification=True
        # model_cls=TfModelSingleton(model_path=model_path_classification)
        model_cls = tensorflow_load_model_cls(model_path=model_path_classification)
        # face_cascade=cv2.CascadeClassifier("./weights/haarcascade_frontalface_default.xml")
        # face_cascade=CascadeSingleton("./weights/haarcascade_frontalface_default.xml")
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path_classification}")
        st.error(ex)

if not classification:
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    source_img = None
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    results = model([uploaded_image])
                    predictions_pandas = results.pandas().xyxy[0]
                    predictions_pandas = predictions_pandas[[predictions_pandas.columns[-1]] + list(predictions_pandas.columns[:-1])]
                    # boxes = res[0].boxes
                    # res_plotted = res[0].plot()[:, :, ::-1]

                    annotated_image = results.render()[0] 
                    st.image(annotated_image, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            st.dataframe(predictions_pandas)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")

    elif source_radio == settings.VIDEO:
        print(classification)
        helper.play_stored_video(conf=0, model=model)
    else:
        st.error("Please select a valid source type!")
else:
    
    source_img = st.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    if source_img is not None:
        
        uploaded_image = PIL.Image.open(source_img)
        desired_width = 300
        desired_height = 300

        # Resize the image
        resized_image = uploaded_image.resize((desired_width, desired_height))
        st.image(resized_image, caption="Uploaded Image")
       
        # list_faces=detect_faces(uploaded_image,face_cascade)
        # print(model_cls)
        result=predict_image(uploaded_image,model_cls)
        if result is not None:
            st.write(f"Result:")
            print(result)

            st.dataframe(pd.DataFrame.from_dict(result))

            # st.write(f"{result}")
        else:
            st.write(f"No Face Detections")
        
    
