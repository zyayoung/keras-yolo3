# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys

import os

from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2

import pickle
import gzip

from utils.nms import nms
from mht import MHT

class MultiTracker:
    def __init__(self, video_path, output_path="", score=0.1, nms_threshold=0.45):
        self.score = score
        self.nms_threshold = nms_threshold

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / 1000, 1., 1.)
                      for x in range(1000)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        self.vid = cv2.VideoCapture(video_path)
        if not self.vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC    = int(self.vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = self.vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            self.out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        else:
            self.out = None
        
        bbox_path ='.'.join(video_path.split('.')[:-1])+'.pkl.gz'
        with gzip.open(bbox_path, "rb") as f:
            self.bbox_history = pickle.load(f)
        
        self.mot_tracker = MHT(obj_num=24)

    def process_frame(self):
        out_boxes, out_scores, out_classes = self.bbox_history.pop(0)

        dets = np.hstack([out_boxes, np.expand_dims(out_scores, -1)])
        keep = nms(dets, self.nms_threshold)
        out_boxes = out_boxes[keep]
        out_scores = out_scores[keep]
        out_classes = out_classes[keep]
        dets = dets[keep]

        keep = out_scores > self.score
        out_boxes = out_boxes[keep]
        out_scores = out_scores[keep]
        out_classes = out_classes[keep]
        dets = dets[keep]
        return {
            'rois': out_boxes,
            'scores': out_scores,
            'class_ids': out_classes,
        }
    
    def run(self):
        data = [self.process_frame() for _ in range(len(self.bbox_history))]
        self.mot_tracker.iterTracking(data)
        return
        while True:
            return_value, frame = self.vid.read()
            if not return_value:
                break
            image = Image.fromarray(frame)
            result = np.asarray(image)
            
            if self.out:
                self.out.write(result)
            else:
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if self.out:
            self.out.release()

def track_video(video_path, output_path=""):
    mt = MultiTracker(video_path, output_path)    
    mt.run()
    
    

