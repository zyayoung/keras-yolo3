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
from sort import Sort

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
        
        self.mot_tracker = Sort(max_age=12, min_hits=3)

    def detect_frame(self, image):
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

        trackers, lost, rets = self.mot_tracker.update(dets)
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32')
        )
        thickness = (image.size[0] + image.size[1]) // 1000

        for i, d in list(enumerate(trackers)):
            box = d[:4]
            ins_id = int(d[4]+0.5)
            label = '{}'.format(ins_id)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[ins_id])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[ins_id])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        for i, d in list(enumerate(lost)):
            box = d[:4]
            ins_id = int(d[4]+0.5)
            label = '{}'.format(ins_id)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(127, 127, 127))
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(127, 127, 127))
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image, out_boxes, out_scores, out_classes
    
    def run(self):
        while True:
            return_value, frame = self.vid.read()
            if not return_value:
                break
            image = Image.fromarray(frame)
            image, out_boxes, out_scores, out_classes = self.detect_frame(image)
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
    
    

