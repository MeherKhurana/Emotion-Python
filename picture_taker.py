import cv2
import time
import numpy as np
from keras.preprocessing import image
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

import face_recognition  # Import face_recognition library
import tensorflow as tf

# Define the model architecture
model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

# Load the saved model weights
model.load_weights('FER_model.h5')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to detect faces in an image
def detect_faces(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    return face_locations

# Function to zoom in on the detected face in the image
def zoom_in_on_face(image_path, face_location):
    img = cv2.imread(image_path)
    top, right, bottom, left = face_location

    # Extract the face region
    face = img[top:bottom, left:right]

    # Zoom in by resizing the face region
    zoomed_face = cv2.resize(face, (48, 48))

    # Save the zoomed-in face as an image file (e.g., 'zoomed_face.jpg')
    cv2.imwrite('zoomed_face.jpg', zoomed_face)

    print("Zoomed-in face saved successfully.")

# Modify the take_picture function to include face recognition and zoom in
def take_picture():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    time.sleep(2)

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        return

    # Save the captured frame as an image file (e.g., 'my_picture.jpg')
    cv2.imwrite('my_picture.jpg', frame)

    # Detect faces in the captured frame
    face_locations = detect_faces('my_picture.jpg')

    if face_locations:
        # Zoom in on the first detected face
        zoom_in_on_face('my_picture.jpg', face_locations[0])
    else:
        print("No faces detected.")

    cap.release()
    print("Picture taken and face zoomed in successfully.")


# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48), grayscale=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array


# Function to predict emotion from an image
def predict_emotion():
    # Path to the test image you want to use
    test_image_path = "zoomed_face.jpg"  # Replace with the actual path

    # Preprocess the image
    preprocessed_image = preprocess_image(test_image_path)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Mapping of class index to emotion label
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    # Get the emotion label for the predicted class
    predicted_emotion = emotion_labels[predicted_class_index]

    print("Predicted Emotion:", predicted_emotion)

while True:
    take_picture()
    time.sleep(1)
    predict_emotion()

# Release the OpenCV windows
cv2.destroyAllWindows()