import cv2 # opencv2 package for python.
import torch
from torch import hub # Hub contains other models like FasterRCNN
import numpy as np

"""
The Function below oracestrates the entire operation and performs the real-time parsing for video stream.
"""

class obj_detection:

    model = torch.hub.load( \
                    'ultralytics/yolov5', \
                    'yolov5s', \
                    pretrained=True)

    def __call__(self):
        cap = cv2.VideoCapture(0) # 0 means read from local camera.
        success, image = cap.read()

        while cap.isOpened():
            results = self.score_frame(image)
            image = self.plot_boxes(results, image)

            cv2.imshow('obj detection', image)
            success, image = cap.read()

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """
    def score_frame(self, frame):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        frame = [torch.tensor(frame)]
        frame = [t.numpy() for t in frame]
        results = self.model(frame)
        labels = results.xyxyn[0][:, -1].numpy()
        cord = results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    """
    The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
    """
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2: 
                continue
            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)
            bgr = (0, 255, 0) # color of the box
            classes = self.model.names # Get the name of label index
            label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
            cv2.rectangle(frame, \
                        (x1, y1), (x2, y2), \
                        bgr, 2) #Plot the boxes
            cv2.putText(frame,\
                        classes[labels[i]], \
                        (x1, y1), \
                        label_font, 0.9, bgr, 2) #Put a label over box.
            return frame

obj = obj_detection()
obj()


