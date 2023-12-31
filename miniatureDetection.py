from typing import Any
import torch
import numpy as np
import cv2
from time import time


class MiniatureDetection:

    def __init__(self, capture_index, model_name):
        
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device: ", self.device)


    def load_model(self, model_name):
        '''
        Loads YOLOv5 model from pytorch hub.
        :return: Trained pytorch model.
        '''
        model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name, force_reload=True)
        return model
    
    def score_frame(self, frame):
        '''
        Takes a frame as input and scores it with yolov5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: labels and coordinates of miniatures detected by model.
        '''
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    
    def class2label(self, x):
        '''
        Convert the label value to the corresponding label string.
        :param x: label numerical value.
        :return: corresponding label string.
        '''
        return self.classes[int(x)]
    
    def drawBoxes(self, results, frame):
        '''
        Draws the bounding boxes on the frame and writes the corresponding label.
        :param results: labels and coordinates predicted by the model in the current frame.
        :param frame: current scored frame.
        :return: frame with bounding boxes and labels.
        '''
        labels, coord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[0], frame.shape[1]

        for i in range(n):
            row = coord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class2label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
    
    def __call__(self):
        '''
        On class call, runs the loops to read the video frame by frame and write output in a new file
        :return: void
        '''
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Errore: WebCam non trovata")
            exit(1)

        while True:
            ret, frame = cam.read()
            assert ret

            frame = cv2.resize(frame, (640,640))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.drawBoxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)

            cv2.putText(frame, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow("Miniature Detection", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        cam.release()


if __name__ == "__main__":
    miniDetector = MiniatureDetection(capture_index=0, model_name='best.pt')
    miniDetector()