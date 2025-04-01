***Study of Techniques for Optimal Image Classification***
***Abstract***
Computer Vision is a discipline of Computer Science which allows media to processed and turned into useful information.  Applications for Computer Vision can be used in image classification to identify cars, people, faces, and various objects.  Computer Vision may also allow images to be altered so useful information can be extracted and placed on to an image directly.  For Computer Vision to be as accurate and efficient as it is, several techniques are to be used.  Techniques such as Neural Networks, Deep Learning, Noise Reduction, Smoothing, Gaussian Blur, Edge Detection, and many others, can enhance images and other forms of media for optimal image classification.

***1.	Background and Significance***
Background:  For Edge detection, a 2D array of an image is constructed showing the magnitude of individual pixels which can be compared to neighboring pixels.  The difference of pixel magnitudes known as a gradient are then applied with operators known as Prewitt, Sobel, Laplacian, Canny operators.  The Prewitt operator is used by directly comparing neighboring points to find edges that are horizontal or vertical.  The Sobel operator compares a center pixel to surrounding pixels.  The Laplacian operator is a combination of the Sobel and Prewitt operators.  Lastly, there is also a more thorough and complex operator known as the Canny operator is more complex and thorough operator that uses thresholds for image edge detection.  
Significance:  With a way to classify images accurately and efficiently, documentation will be assembled to explain and compare various techniques focusing on speed and accuracy.  The significance of comparing Image classification techniques will allow for tasks to be automated and completed at their highest potential.  Applications of this documentation will allow for the ability to identify what is in an individual surrounding to determine their location for navigation purposes, as well as the ability to learn about the terminology of different types of architecture for educational purposes.  

***2.	Methodology***
To develop this study, OpenCV will be used as a foundation to read in images, convert them into a working format, process them, and output the resulting classifications and their measurements of efficiency and accuracy.  Specifically, OpenCV will allow the use of the You Only Look Once, or the YOLO algorithm for image classification via a neural network.

***3.	Statement of Qualification***
This study can be accomplished as I’m a senior Computer Science student with course work in data mining, calculus, and various programming languages, all of which have provided me with the ability to use a foundation of knowledge to efficiently solve problems.

***4.	Expected Outcomes***
The results of this study will produce documentation that highlights the most efficient way to classify images, and possible use cases.  To start, an unseen set of images will be classified before and after undergoing edge detection, noise reduction, gaussian blur, smoothing, deep learning, or neural network techniques.  Images will be classified using these techniques, or a combination of them.  This study will observe the accuracy, efficiency, and F1 Score of the classified objects, as well as the time needed for a technique to return results.

***Bibliography***

[1]
G. T. Shrivakshan, “A Comparison of various Edge Detection Techniques used in Image Processing,” vol. 9, no. 5, 2012.
[2]
F. Wang and M. Zhang, “Deep Learning-Based Edge Detection Algorithm for Noisy Images,” in Proceedings of the 2023 6th International Conference on Artificial Intelligence and Pattern Recognition, in AIPR ’23. New York, NY, USA: Association for Computing Machinery, Jun. 2024, pp. 465–472. doi: 10.1145/3641584.3641653.
[3]
L. Team, “Edge Detection Using OpenCV | LearnOpenCV #.” Accessed: Mar. 30, 2025. [Online]. Available: https://learnopencv.com/edge-detection-using-opencv/
[4]
C. Zhao, S. Pan, and W. Wang, “Improved Canny Edge Detection Algorithm for Noisy Images,” in Proceedings of the 4th International Conference on Artificial Intelligence and Computer Engineering, in ICAICE ’23. New York, NY, USA: Association for Computing Machinery, May 2024, pp. 84–89. doi: 10.1145/3652628.3652642.
[5]
“OpenCV: Canny Edge Detection.” Accessed: Mar. 31, 2025. [Online]. Available: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
[6]
“OpenCV: Image Gradients.” Accessed: Mar. 31, 2025. [Online]. Available: https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
[7]
“What Are The Applications of Edge Detection?” Accessed: Mar. 30, 2025. [Online]. Available: https://www.plugger.ai/blog/what-are-the-applications-of-edge-detection
[8]
J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” May 09, 2016, arXiv: arXiv:1506.02640. doi: 10.48550/arXiv.1506.02640.
![image](https://github.com/user-attachments/assets/48033686-7119-4970-9bb4-e30afefdd50b)
