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