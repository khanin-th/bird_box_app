import streamlit as st
import pickle
import numpy as np
# model = pickle.load(open('final_model.pkl','rb'))
from PIL import Image

import os
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

import sys
sys.path.insert(0, r'C:\Users\pui_s\Downloads\TensorflowObjectDetection\TFODCourse\tfod')
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
    eval_config: an eval config containing the keypoint edges

    Returns:
    a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list
	
	
	
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(r'C:\Users\pui_s\Documents\BirdBox_helping_drone_detect_bird\trained_ssd_mobnet\pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(r'C:\Users\pui_s\Documents\BirdBox_helping_drone_detect_bird\trained_ssd_mobnet\checkpoint', 'ckpt-0')).expect_partial()

def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn

detect_fn = get_model_detection_function(detection_model)

label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)


st.set_page_config(
    page_title="Object Detection App",
    layout="centered",
    initial_sidebar_state="expanded",
)
def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image,
              use_column_width=True)
        st.write("type = ", type(uploaded_file))
        st.write("value = ", uploaded_file)
        
    if st.button("Find Bird"):
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0].numpy()
            keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.30,
              agnostic_mode=False,
              keypoints=keypoints,
              keypoint_scores=keypoint_scores,
              keypoint_edges=get_keypoint_tuples(configs['eval_config']))
        
        st.image(image_np_with_detections, use_column_width=True)


# def predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight):
#     input=np.array([[Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight]]).astype(np.float64)
#     prediction = model.predict(input)
#     #pred = '{0:.{1}f}'.format(prediction[0][0], 2)
#     return int(prediction)


# def main():
#     #st.title("Abalone Age Prediction")
#     html_temp = """
#     <div style="background:#025246 ;padding:10px">
#     <h2 style="color:white;text-align:center;">Abalone Age Prediction ML App </h2>
#     </div>
#     """
#     st.markdown(html_temp, unsafe_allow_html = True)

#     # input
#     Length = st.text_input("Length")
#     Diameter = st.text_input("Diameter")
#     Height = st.text_input("Height")
#     Whole_weight = st.text_input("Whole weight")
#     Shucked_weight = st.text_input("Shucked weight")
#     Viscera_weight = st.text_input("Viscera weight")
#     Shell_weight = st.text_input("Shell weight")

#     safe_html ="""  
#       <div style="background-color:#80ff80; padding:10px >
#       <h2 style="color:white;text-align:center;"> The Abalone is young</h2>
#       </div>
#     """
#     warn_html ="""  
#       <div style="background-color:#F4D03F; padding:10px >
#       <h2 style="color:white;text-align:center;"> The Abalone is middle aged</h2>
#       </div>
#     """
#     danger_html="""  
#       <div style="background-color:#F08080; padding:10px >
#        <h2 style="color:black ;text-align:center;"> The Abalone is old</h2>
#        </div>
#     """

#     # output
#     if st.button("Predict the age"):
#         output = predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight)
#         st.success('The age is {}'.format(output))

#         if output == 1:
#             st.markdown(safe_html,unsafe_allow_html=True)
#         elif output == 2:
#             st.markdown(warn_html,unsafe_allow_html=True)
#         elif output == 3:
#             st.markdown(danger_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
