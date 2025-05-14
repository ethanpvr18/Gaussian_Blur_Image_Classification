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


font_size = 0.35
thickness = 1

for image_path in images:
    gaussianBlurKernel = 1    # Kernal Size
    counter = 0
    numClass = []
    first = 0
    kernelSizes = []
    objectConfidences = []
    allConfidences = []
    
    while True:
        # Load image
        if isinstance(image_path, str) and os.path.isfile(image_path):
            img = cv.imread(image_path)
            modified = cv.GaussianBlur(img, (gaussianBlurKernel,gaussianBlurKernel), 0)
    
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

            cv.imshow('blob', r)
            text = f'Blob shape={blob.shape}'
            cv.displayOverlay('blob', text)
            cv.waitKey(1)
        
            # Perform forward pass
            net.setInput(blob)
            t0 = time.time()
            outputs = net.forward(ln)
            t = time.time()
        
            # Display processing time on the image using putText
            processing_time_text = f'Forward propagation time: {t - t0:.2f} sec'
            cv.putText(modified, processing_time_text, (15, 15), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), thickness)
        
            # Extract bounding boxes, class IDs, and confidence scores
            boxes = []
            confidences = []
            class_ids = []

            avgConfidencesPerBlur = 0
    
            h, w = modified.shape[:2]
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    avgConfidencesPerBlur += confidence
        
                    if confidence > 0.5:  # Confidence threshold
                        box = detection[0:4] * np.array([w, h, w, h])
                        (center_x, center_y, width, height) = box.astype("int")
        
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
        
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            cv.displayOverlay('window', f'forward propagation time={t-t0}')
            cv.imshow('window',  img)
            cv.waitKey(0)
            cv.destroyAllWindows()

            avgConfidencesPerBlur = avgConfidencesPerBlur/len(output)
            allConfidences.append(avgConfidencesPerBlur)

            # Apply Non-Maximum Suppression (NMS)
            indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
            # Draw bounding boxes and labels
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                    color = (0, 255, 0)
        
                    cv.rectangle(modified, (x, y), (x + w, y + h), color, thickness)
                    cv.putText(modified, label, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
            if first == 0 and len(indices) > 0:
                first = len(indices)

            kernelSizes.append(gaussianBlurKernel)

            if first != 0:
                numClass.append((len(indices)/first)*100)
            else:
                numClass.append(0)
        
            if len(indices) == 0:
                plt.plot(kernelSizes, allConfidences)
                break
        
            gaussianBlurKernel += 2
            counter += 1


plt.xlabel("Kernel Size of Gaussian Blur")
plt.ylabel("Confidence Level")
plt.legend()
plt.grid(axis='both', which='minor')
plt.title("Impact of Gaussian Blur on Image Classification Confidence")
plt.savefig(f'plot_confidences.pdf')
