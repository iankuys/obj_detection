import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=1,
                            min_detection_confidence= 0.5,
                            min_tracking_confidence=0.8,
                            model_name='Cup') as objectron:

    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        # mediapipe only works with rgb images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = True
        results = objectron.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

        end = time.time()
        totalTime = end - start
        
        cv2.imshow('Mediapipe Objectron', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

