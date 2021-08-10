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
import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

import tensorflow as tf

import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

st.set_page_config(
    page_title="Bird Detection App",
    layout="centered",
    initial_sidebar_state="expanded",
)

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
	
option = st.selectbox('Select the model', ('Centernet2', 'Centernet', 'SSD', 'EfficientDet'))

if option is not None:
    if option == 'Centernet2':
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(r'trained_center_mobnet\pipeline.config')
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(r'trained_center_mobnet\checkpoint', 'ckpt-0')).expect_partial()

    elif option == 'Centernet':
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(r'trained_center_mobnet2\pipeline.config')
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(r'trained_center_mobnet2\checkpoint', 'ckpt-0')).expect_partial()
    
    
    elif option == 'SSD':
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(r'trained_ssd_mobnet\pipeline.config')
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(r'trained_ssd_mobnet\checkpoint', 'ckpt-0')).expect_partial()
    
    elif option == 'EfficientDet':
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(r'trained_EfficientDet_D0\pipeline.config')
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(r'trained_EfficientDet_D0\checkpoint', 'ckpt-0')).expect_partial()
    


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

# choosing content
option_mode = st.selectbox('Select the content', ('Image', 'Video'))

def main():
    # image 
    if option_mode == 'Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image,
                use_column_width=True)
            # st.write("type = ", type(uploaded_file))
            # st.write("value = ", uploaded_file)
            detection_threshold = st.slider(label="set detection threshold: ",
                                    min_value=0.,
                                    max_value=1.,
                                    value=0.3,
                                    step=0.05
                                    )
            
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
                min_score_thresh=detection_threshold,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=get_keypoint_tuples(configs['eval_config']))
            
            st.image(image_np_with_detections, use_column_width=True)

    # Video process
    elif option_mode == 'Video':
        uploaded_file_video = st.file_uploader("Choose a video...", type=["mp4", "mpeg", 'mov', 'gif'])
        if uploaded_file_video is not None:
            st.video(uploaded_file_video)

            vid = uploaded_file_video.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_file_video.read()) # Saves uploaded video to disk so that you can use cv2.VideoCapture()
            video = cv2.VideoCapture(vid)
            video_fps = video.get(cv2.CAP_PROP_FPS)
            st.write(f"default fps: {video_fps}")
            video.release()

            # user input
            frame_rate = st.slider(label="set output fps: ",
                                    min_value=1,
                                    max_value=100,
                                    value=int(video_fps),
                                    step=1
                                    )
            detection_threshold = st.slider(label="set detection threshold: ",
                                    min_value=0.,
                                    max_value=1.,
                                    value=0.3,
                                    step=0.05
                                    )

        
        if st.button("Find Bird"):
            
            video_reader = imageio.get_reader(vid)

            # name, and output fps can be controlled here
            video_writer = imageio.get_writer(vid.split('.')[0]+r'_annotated.mp4', fps=frame_rate)

            # looping to process each frame
            t0 = datetime.now()
            n_frames = 0

            for frame in video_reader:
                image_np = np.array(frame)
                n_frames = n_frames+1
                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                detections, predictions_dict, shapes = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                            image_np_with_detections,
                            detections['detection_boxes'],
                            detections['detection_classes']+label_id_offset,
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=5,
                            # can change here
                            min_score_thresh=detection_threshold,
                            # end here
                            agnostic_mode=False)


                # video_writer.append_data(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
                video_writer.append_data(image_np_with_detections)
                
            fps = n_frames/(datetime.now()-t0).total_seconds()
            st.write("Frames processed: %s, Processing Speed:%s fps"%(n_frames, fps))

            # cleanup
            video_writer.close()

            # displaying the processed video
            st.video(vid.split('.')[0]+r'_annotated.mp4')

if __name__=='__main__':
    main()
