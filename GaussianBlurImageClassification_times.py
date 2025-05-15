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


font_size = 0.35
thickness = 1

# Load YOLO model
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# Get output layer names
layer_names = net.getLayerNames()
ln = net.getUnconnectedOutLayersNames()

# Load COCO class labels
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
                
for image_path in images:
    gaussianBlurKernel = 3    # Kernal Size
    counter = 0
    numClass = []
    first = 0
    kernelSizes = []
    objectConfidences = []
    allConfidences = []
    allTimes = []
    
    while True:
        # Load image
        if isinstance(image_path, str) and os.path.isfile(image_path):
            img = cv.imread(image_path)
            modified = cv.GaussianBlur(img, (gaussianBlurKernel,gaussianBlurKernel), 0)
    
            # Create blob from image
            blob = cv.dnn.blobFromImage(modified, 1/255.0, (416, 416), swapRB=True, crop=False)
            r = blob[0, 0, :, :]
        
            # Perform forward pass
            net.setInput(blob)
            t0 = time.time()
            outputs = net.forward(ln)
            t = time.time()
        
            # Display processing time on the image using putText
            processing_time_text = t - t0:.2f
            allTimes.append(processing_time_text)

            kernelSizes.append(gaussianBlurKernel)
        
            if len(indices) == 0 or gaussianBlurKernel > 100:
                # print(f'Image Path: {str(image_path)} : Time: {t - t0:.2f} sec')
                plt.plot(kernelSizes, allTimes, label=str(image_path))
                break
        
            gaussianBlurKernel += 2
            counter += 1
            

plt.xlabel("Kernel Size of Gaussian Blur")
plt.ylabel("Time to Classify")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='both', which='minor')
plt.title("Impact of Gaussian Blur on Image Classification")
plt.savefig(f'plot_times.pdf')
