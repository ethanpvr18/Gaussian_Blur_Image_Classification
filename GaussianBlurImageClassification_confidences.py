from itertools import count

import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Pull images to be tested
input_folder = 'images'
output_dir = "downloaded_images"
os.makedirs(output_dir, exist_ok=True)
images = []

numImages = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        if numImages < 5:
            image_path = os.path.join(input_folder, filename)
            images.append(image_path)
            numImages += 1

for image_path in images:
    blur = 3    # Kernal Size
    numClass = []
    first = 0
    kernelSizes = []
    allConfidences = []
    
    while True:
        # Load image
        if isinstance(image_path, str) and os.path.isfile(image_path):
            img = cv.imread(image_path)
            modified = cv.GaussianBlur(img, (blur,blur), 0)
    
            # Load YOLO model
            net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    
            # Get output layer names
            layer_names = net.getLayerNames()
            ln = net.getUnconnectedOutLayersNames()
    
            # Load COCO class labels
            with open('coco.names', 'r') as f:
                classes = f.read().strip().split('\n')
        
            # Create blob from image
            blob = cv.dnn.blobFromImage(modified, 1/255.0, (416, 416), swapRB=True, crop=False)
            r = blob[0, 0, :, :]
        
            # Perform forward pass
            net.setInput(blob)
            t0 = time.time()
            outputs = net.forward(ln)
            t = time.time()

            avgConfidencesPerBlur = 0
    
            for output in outputs:
                count = 0
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    avgConfidencesPerBlur += confidence
                    count += 1
        
                avgConfidencesPerBlur = avgConfidencesPerBlur/count
                allConfidences.append(avgConfidencesPerBlur)
        
            if first == 0 and len(indices) > 0:
                first = len(indices)

            kernelSizes.append(blur)

            if first != 0:
                numClass.append((len(indices)/first)*100)
            else:
                numClass.append(0)
        
            if len(indices) == 0:
                plt.plot(kernelSizes, allConfidences, label=f'{image_path}')
                break
        
            blur += 2

plt.xlabel("Kernel Size of Gaussian Blur")
plt.ylabel("Confidence Level")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='both', which='minor')
plt.title("Impact of Gaussian Blur on Image Classification Confidence")
plt.savefig(f'plot_confidences.pdf')
