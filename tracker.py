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
    def __init__(self, video_path, output_path="", score=0.01, nms_threshold=0.65):
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
        obj_num = len(self.process_frame(*self.bbox_history[0])['scores'])
        print("obj_num:", obj_num)
        self.mot_tracker = MHT(obj_num=obj_num)

    def process_frame(self, out_boxes, out_scores, out_classes):
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
    
    def process_frames(self):
        return [self.process_frame(*bbox) for bbox in self.bbox_history]
    
    def visualize(self, img, rois):
        for roi in rois:
            cv2.rectangle(img, (roi[1], roi[0]), (roi[3], roi[2]), (255, 0, 0), 5)
        return img
    
    def run(self):
        rois_total = self.mot_tracker.iterTracking(self.process_frames())
        rois_total = np.array(rois_total, dtype=int).transpose(1, 0, 2)
        print(rois_total)
        for rois in rois_total:
            return_value, frame = self.vid.read()
            if not return_value:
                break
            image = Image.fromarray(frame)
            result = np.asarray(image)
            result = self.visualize(result, rois)
            
            if self.out:
                self.out.write(result)
            else:
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        if self.out:
            self.out.release()

def track_video(video_path, output_path=""):
    mt = MultiTracker(video_path, output_path)    
    mt.run()
    
    

