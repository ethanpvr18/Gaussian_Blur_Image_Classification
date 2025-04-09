from itertools import count

import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt

font_size = 0.35
thickness = 1
blur = 3
counter = 0
numClass = []
first = 0
kernelSizes = []
images = ['city_street.jpg']

for image_path in images:
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
        
            # Display processing time on the image using putText
            processing_time_text = f'Forward propagation time: {t - t0:.2f} sec'
            cv.putText(modified, processing_time_text, (15, 15), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), thickness)
        
            # Extract bounding boxes, class IDs, and confidence scores
            boxes = []
            confidences = []
            class_ids = []
    
            h, w = modified.shape[:2]
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
        
                    if confidence > 0.5:  # Confidence threshold
                        box = detection[0:4] * np.array([w, h, w, h])
                        (center_x, center_y, width, height) = box.astype("int")
        
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
        
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
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
        
            if first == 0:
                first = len(indices)
        
            kernelSizes.append(blur)
            numClass.append((len(indices)/first)*100)
        
            if len(indices) == 0:
                # print(numClass)
                plt.plot(kernelSizes, numClass)
                
                break
        
            blur += 2
            counter += 1
            
    plt.xlabel("Kernel Size of Gaussian Blur")
    plt.ylabel("Percent of Objects Classified")
    plt.title("Impact of Gaussian Blur on Image Classification")
    plt.savefig(f'plot.pdf')
