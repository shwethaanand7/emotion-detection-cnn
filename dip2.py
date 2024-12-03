import cv2
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,Input
import numpy as np

# Explicitly specify TensorFlow version to avoid compatibility issues
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Disabling eager execution to avoid attribute errors

# Load the model architecture from JSON file
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load model weights
model.load_weights("emotiondetector.h5")

# Path to Haar cascade XML file
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Access the webcam
webcam = cv2.VideoCapture(0)

# Labels for emotion classes
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read frame from webcam
    i, im = webcam.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        # Iterate through detected faces
        for (p, q, r, s) in faces:
            # Extract face region
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            
            # Resize face image to match model input shape
            image = cv2.resize(image, (48, 48))
            
            # Extract features and preprocess
            img = extract_features(image)
            
            # Make prediction
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Display predicted emotion label
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display output frame
        cv2.imshow("Output", im)
        
        # Check for key press to exit (ESC key)
        key = cv2.waitKey(1) 
        if key == 27:  # ESC key
            break
    except cv2.error:
        pass

# Release webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
