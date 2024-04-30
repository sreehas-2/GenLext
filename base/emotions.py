import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from scipy.spatial import distance
from imutils import face_utils
import numpy as np


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

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 100


#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("base/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("base/haarcascade_eye.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear


#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def predict_emotion(input_screenshot_array):
    print("inside pred fun")
    check=0
    gray_screenshot = cv2.cvtColor(input_screenshot_array, cv2.COLOR_BGR2GRAY)

    # Find faces in the input screenshot
    faces = face_cascade.detectMultiScale(gray_screenshot, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return [4]
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray_screenshot[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            #Detect eyes in face
            eyes = eye_cascade.detectMultiScale(roi_gray)
            ear=0
            cnt=0
            for (ex, ey, ew, eh) in eyes:
                # Calculate aspect ratio of the detected eye
                
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                ear =ear+ eye_aspect_ratio([(ex, ey), (ex + ew, ey), (ex, ey + eh), (ex + ew, ey + eh), (ex + ew//2, ey + eh//2), (ex + ew//2, ey)])
                
                # print(ex,ey,ew,eh)
                cnt=cnt+1
            print(ear)
            if(ear < EYE_ASPECT_RATIO_THRESHOLD):
                check=1
            print("eyescount ")
            print(check)
        result=[max_index,check]
        print("h")
        print(result[0])
        return result
    