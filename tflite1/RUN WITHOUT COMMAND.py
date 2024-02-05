# Import necessary packages
import os
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pygame  # Added for alarm
from tensorflow.lite.python.interpreter import Interpreter

# Define VideoStream class to handle streaming of video from webcam in a separate processing thread
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Initialize pygame for playing alarm sound
pygame.init()
pygame.mixer.init()

# Load an alarm sound file (replace 'path/to/alarm.wav' with the actual path to your alarm sound file)
alarm_sound = pygame.mixer.Sound('C:\\Users\\HP\\Desktop\\CIGARETTE-SMOKING-DETECTION-IN-REAL-TIME\\tflite1\\mixkit-emergency-alert-alarm-1007.wav')

# Hardcoded values
MODEL_NAME = "custom_model_lite"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = 0.5
resW, resH = "1280x720".split('x')
imW, imH = int(resW), int(resH)
use_TPU = False  # Change to True if you want to use Edge TPU

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Initialize TensorFlow Lite interpreter
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load labels from label map file
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize frame_rate_calc
frame_rate_calc = 0

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Initialize frame_rate_calc and freq
frame_rate_calc = 0
freq = cv2.getTickFrequency()

while True:
    t1 = cv2.getTickCount()

    frame1 = videostream.read()

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if 0.5 <= scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Trigger alarm for a specific object (e.g., person) with high confidence
            if object_name == 'smoke' and scores[i] > 0.8:
                pygame.mixer.Sound.play(alarm_sound)

    cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):  # Press 's' to stop the alarm
        pygame.mixer.stop()

cv2.destroyAllWindows()
videostream.stop()
