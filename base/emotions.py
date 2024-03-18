import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('base/model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def predict_emotion(input_screenshot_array):
    print("inside pred fun")
    gray_screenshot = cv2.cvtColor(input_screenshot_array, cv2.COLOR_BGR2GRAY)

    # Find faces in the input screenshot
    face_cascade = cv2.CascadeClassifier('base\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_screenshot, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return 4
    else:
        # Store the predicted emotions for each face
        face_emotions = []

        for (x, y, w, h) in faces:
            roi_gray = gray_screenshot[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
        return max_index 


