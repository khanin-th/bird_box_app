# bird_box_app
Object detection model for bird

## Setup a new virtual environment to run this app (if you never had TensorFlow Object Detection API installed)
### Step 1: 
Create a new directory to work in (for example here I will create a folder called "bird_box" in Downloads directory) and navigate to the new directory <br>
```
mkdir C:\Users\pui_s\Downloads\bird_box && cd C:\Users\pui_s\Downloads\bird_box
```

### Step 2:
Create a new virtual environment (so that your main python won't be disturbed), and name it, which is `bird_od` in this example
```
python -m venv bird_od
```

### Step 3:
Activate the virtual environment to work in. Now you will notice a bracket pre-appended to what being displayed in the command line windows as shown below
```
.\bird_od\Scripts\activate
```

and for mac os
```
source bird_od/bin/activate
```

![cmd1](img/cmd1.png)

### Step 4:
Create 2 new directories to store `TensorFlow Object Detection API` and its dependency `prtobuf`, here I am creating a new directory called "Tensorflow" and 2 new sub-directories for such purposes
```
mkdir Tensorflow\models;Tensorflow\protoc
```

### Step 5:
Clone TensorFlow Object Detection models
```
git clone https://github.com/tensorflow/models Tensorflow\models
```

### Step 6:
Download a zip file from https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-win64.zip for protobuf and uncompress it inside protoc folder, and navigate back to the main `bird_box` directory
```
move C:\Users\pui_s\Downloads\protoc-3.17.3-win64.zip Tensorflow\protoc && cd Tensorflow\protoc && tar -xf protoc-3.17.3-win64.zip
cd ..\..
```

### Step 7:
Copy `protoc.exe` to Scripts folder of the vistual environment
```
copy Tensorflow\protoc\bin\protoc.exe bird_od\Scripts
```

### Step 8:
Navigate to `Tensorflow\models\research` and install the required `object_detection` module
```
cd Tensorflow\models\research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install
```

### Step 9:
Install tf-slim
```
cd ..\..\..\Tensorflow/models/research/slim && pip install -e .
```

### Step 10:
Install `tensorflow` or `tensorflow-gpu` depending on your machine via pip
```
pip install tensorflow
```

### Step 11:
Test whether the installation has been done sucessfully. Note that you might need to pip install any modules that you are lacking, i.e. `pip install matplotlib`, `pip install pyyaml`.
```
python ..\object_detection/builders/model_builder_tf2_test.py
```

Sucessful installtion will display `OK (skipped=1)` as shown below
![sucessful installtion](img/cmd2.png)


### Step 12:
Clone this repository to the main `bird_box` folder
```
cd ..\..\..\..
```
Your cmd should now be at `bird_box` directory, next execute the git clone command
```
git clone https://github.com/khanin-th/bird_box_app
move bird_box_app\* 
```

remove the cloned folder
```
rmdir /S bird_box_app
```

### Step 13:
Uncompress the trained models
```
tar -xf trained_center_mobnet.zip
tar -xf trained_ssd_mobnet.zip
tar -xf asset.zip
```

### Step 14:
Install `streamlit` for running the interactive application
```
pip install streamlit
```

### Step 15:
Launch the application
```
streamlit run app.py
```

# Reference:
1. S. Deliwala, "Streamlit 101: An in-depth introduction," towardsdatascience.com, 18 November 2019. [Online]. Available: https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2. [Accessed 3 August 2021].
1. T. Morkar, "How to use Streamlit to deploy your ML models wrapped in beautiful Web Apps," towardsdatascience.com, 28 July 2020. [Online]. Available: https://towardsdatascience.com/how-to-use-streamlit-to-deploy-your-ml-models-wrapped-in-beautiful-web-apps-66e07c3dd525. [Accessed 3 August 2021].
1. A. Sharma, "abalone-age-app," github.com, 27 September 2020. [Online]. Available: https://github.com/Apurva-tech/abalone-age-app. [Accessed 3 August 2021].
1. N. Renotte, "Tensorflow Object Detection in 5 Hours with Python | Full Course with 3 Projects," youtube.com, 9 April 2021. [Online]. Available: https://www.youtube.com/watch?v=yqkISICHH-U. [Accessed 30 July 2021].

