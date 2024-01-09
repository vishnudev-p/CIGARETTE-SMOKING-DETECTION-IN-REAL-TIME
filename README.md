# CIGARETTE-SMOKING-DETECTION-IN-REAL-TIME
Smoking detection is a critical aspect of public health and safety, and this project aims to address it through the use of deep learning techniques. The model developed here is trained to identify instances of cigarette smoking in images, providing a tool for applications such as survellience
## Overview

The model is trained using a dataset sourced from [Roboflow](https://roboflow.com/), which provides a carefully annotated dataset for cigarette smoking detection. The dataset is preprocessed and includes diverse examples of smoking scenarios, making the model robust to various real-world conditions.

## Google Colab Training

The model training process is documented in the `train_model.ipynb` notebook, hosted on Google Colab. Training on Colab enables efficient GPU acceleration, allowing for faster model convergence. To train the model, follow the provided link (link-https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=GSJ2wgGCixy2) to the notebook.

## TensorFlow Lite Conversion

After training, the model is converted to TensorFlow Lite for lightweight deployment on edge devices. The conversion script is available in the `https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=GSJ2wgGCixy2` notebook. TensorFlow Lite ensures that the model can run efficiently on resource-constrained devices, making it suitable for real-world applications.

## Running the Model

To use the trained model on your local machne or edge device, follow the instructions in this github repository for windows-'https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/deploy_guides/Windows_TFLite_Guide.md'

1. Clone this repository:

    ```bash
    git clone https://github.com/vishnudev-p/cigarette-smoking-detection-in-real-time.git
    cd cigarette-smoking-detection
    ```

2. Download the TensorFlow Lite model (`model.tflite`) from this repository using clone.

3. After cloning into the repo you can execute the code using this command:
       python TFLite_detection_webcam.py --modeldir=custom_model_lite

5. Demo video for training model is uploaded below you can just watch and understand it:
     'https://www.youtube.com/watch?v=XZ7FYAMCc4M'


