# Video-Emotion-Recognition
Facial expressions play an important role in identifying the emotional state of an individual and help others relate, understand and respond accordingly. Individuals can have different reactions to the same stimuli. This project aims to examine the emotional state of patients experiencing psychosis.The objective is to detect the various emotions of these patients, these emotions might include anger, disgust, fear, joy, sadness, surprise or neutral.  This is achieved by making the patient play a game while their facial expressions are obtained through a live web camera. This is used to monitor and record their emotions as data for medical purpose. Implementing the Haar Algorithm, the frames are cropped and the face alone is procured on which grey scaling and resizing process is carried out. Now the sequence of faces obtained will be used to extract the most necessary features by a CNN - 2D(Convolutional neural network) to extract the most necessary features of each face, which will encode motion and facial expressions to predict emotion. Two sets of data are used as the dataset -  Training set- the algorithm will read, or ‘train’, on this over and over again to try and learn its task, and the Testing set - the algorithm is tested on this data to see how well it works.



Tester files:
These files contain the OpenCV and Haar algorithm along with xlwt.
it ranges from tester.py to tester4.py -> each number indicates an upgraded and more enhanced version of the code. You can run each of them in your terminal to notice the difference.

Emotion_detecion.py:
Emotion_detection2.py is the more enhanced version which provides us with an accuracy greater than 96%, while the first one gives us an accuracy of only 60% even if we run till a 100 epochs. I have included both of them to show the change and the improvement in the output.
