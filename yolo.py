# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from timeit import default_timer as timer
import tensorflow as tf

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, box_iou
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

from munkres import Munkres, DISALLOWED
m = Munkres()

def nms(dets, scores, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = np.zeros_like(scores, dtype=bool)
    while order.size > 0:
        i = order[0]
        keep[i] = 1
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'anchors.txt',
        "classes_path": 'classes.txt',
        "score" : 0.1,
        "iou" : 0.45,
        "model_image_size" : (608, 608),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.prev_bbox = None
        self.prev_velocity = None
        self.prev_bbox_frame_cnt = None
        self.connections = []
        self.instances = []
        self.instance_count = 1
        # print(self.boxes[0])

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    
    @staticmethod
    def bbox_overlap(boxes, query_boxes):
        """
        determine overlaps between boxes and query_boxes
        :param boxes: n * 4 bounding boxes
        :param query_boxes: k * 4 bounding boxes
        :return: overlaps: n * k overlaps
        """
        n_ = boxes.shape[0]
        k_ = query_boxes.shape[0]
        overlaps = np.zeros((n_, k_), dtype=np.float)
        for k in range(k_):
            query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            for n in range(n_):
                iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
                if iw > 0:
                    ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                    if ih > 0:
                        box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                        all_area = float(box_area + query_box_area - iw * ih)
                        overlaps[n, k] = iw * ih / all_area
        return overlaps

    @staticmethod
    def bbox_distance(boxes, query_boxes, area):
        """
        determine distance between boxes and query_boxes
        :param boxes: n * 4 bounding boxes
        :param query_boxes: k * 4 bounding boxes
        :return: distance: n * k overlaps
        """
        ax = (boxes[:,2]+boxes[:,0])/2
        ay = (boxes[:,3]+boxes[:,1])/2
        bx = (query_boxes[:,2]+query_boxes[:,0])/2
        by = (query_boxes[:,3]+query_boxes[:,1])/2
        distance = np.zeros((ax.shape[0], bx.shape[0]), dtype=np.float)
        for i in range(ax.shape[0]):
            for j in range(bx.shape[0]):
                distance[i, j] = np.square(ax[i]-bx[j]) + np.square(ay[i]-by[j])
        distance = np.sqrt(distance/area)
        # distance /= distance.min()
        return distance

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

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

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou, nms=True)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        
        connection = {}
        out_ids = np.zeros(len(out_boxes), int)
        frame_cnt = np.zeros(len(out_boxes))
        out_velocity = np.zeros((len(out_boxes),4))
        if self.prev_bbox is not None:
            # matrix = 1/608*self.bbox_distance(self.prev_bbox, out_boxes, image.width*image.height) \
            #     + 1/self.bbox_overlap(self.prev_bbox, out_boxes)\
            matrix = 1/( \
                self.bbox_overlap(self.prev_bbox+self.prev_velocity, out_boxes) + \
                    np.exp(-61*self.bbox_distance(self.prev_bbox, out_boxes, image.width*image.height))+ \
                    0)# - np.repeat(np.expand_dims(self.prev_bbox_frame_cnt, axis=1)/64, len(out_scores), axis=1) + 1/(out_scores.T+0.5)
            #  - np.repeat(np.expand_dims(self.prev_bbox_frame_cnt, axis=1)/10, len(out_scores), axis=1)
            #  + out_scores.T
            h, w = matrix.shape
            matrix[matrix>1e6]=1e6
            _matrix = np.ones((max(h,w), max(h,w)))*1e6
            _matrix[:h, :w] = matrix.copy()
            # print(_matrix)
            indexes = m.compute(_matrix)
            total = 0
            prev_mask = np.zeros(len(self.prev_bbox), dtype=bool)
            for row, column in indexes:
                if row >= h or column >= w or matrix[row][column]>10:
                    continue
                value = matrix[row][column]
                total += value
                print(f'{row} -> {column} | {value}')
                connection[row] = column
                if self.instances and self.instances[-1][row] > 0:
                    prev_mask[row] = True
                    out_ids[column] = self.instances[-1][row]
                    out_velocity[column] = out_boxes[column] - self.prev_bbox[row]
                    out_velocity[column] = 0.1 * out_velocity[column] + 0.9 * self.prev_velocity[row]
                    out_boxes[column] = 0.5*out_boxes[column] + 0.5*(self.prev_bbox[row] + out_velocity[column])
                    frame_cnt[column] = self.prev_bbox_frame_cnt[row] + 1
            print(f'total cost: {total}')

            for index, box in enumerate(self.prev_bbox):
                if not prev_mask[index] and self.prev_bbox_frame_cnt[index] > 7:
                    out_boxes = np.concatenate([out_boxes, [box+self.prev_velocity[index]]], axis=0)
                    frame_cnt = np.concatenate([frame_cnt, [self.prev_bbox_frame_cnt[index]]], axis=0)
                    out_classes = np.concatenate([out_classes, [0]], axis=0)
                    out_scores = np.concatenate([out_scores, [0]], axis=0)
                    out_velocity = np.concatenate([out_velocity, [0.9*self.prev_velocity[index]]], axis=0)
                    out_ids = np.concatenate([out_ids, [self.instances[-1][index]]], axis=0)

            drop = (out_ids == 0)
            keep = (out_ids > 0)
            iou = self.bbox_overlap(out_boxes[drop], out_boxes[keep]).max(axis=1)
            nms_keep = nms(out_boxes[drop], out_scores[drop], 0.3)
            keep[np.where(drop)[0][(iou<0.2) & nms_keep]] = True
            out_boxes = out_boxes[keep]
            frame_cnt = frame_cnt[keep]
            out_classes = out_classes[keep]
            out_scores = out_scores[keep]
            out_ids = out_ids[keep]
            out_velocity = out_velocity[keep]
        # else:
        #     keep = nms(out_boxes, out_scores, 0.3)
        #     out_boxes = out_boxes[keep]
        #     frame_cnt = frame_cnt[keep]
        #     out_classes = out_classes[keep]
        #     out_scores = out_scores[keep]
        #     out_ids = out_ids[keep]

        self.prev_bbox = out_boxes.copy()
        self.prev_velocity = out_velocity.copy()
        self.prev_bbox_frame_cnt = frame_cnt

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 1000

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            if out_ids[i] == 0:
                out_ids[i] = self.instance_count
                self.instance_count += 1
            ins_id = out_ids[i]
            if frame_cnt[i] < 3:
                ins_id = 0
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

        self.connections.append(connection)
        self.instances.append(out_ids)
        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    channal_diff = np.int32(video_fps)
    history = []
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        # history.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # if len(history)>=channal_diff+1:
        #     frame[:,:,1] = history[-channal_diff-1]
        # if len(history)>=channal_diff*2+1:
        #     frame[:,:,2] = history[-channal_diff*2-1]
        #     history.pop(0)
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        if isOutput:
            out.write(result)
        else:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    yolo.close_session()

