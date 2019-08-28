import os
import numpy as np
import math
import cv2

from treelib import Node, Tree
import networkx as nx

from utils.mwis import MWIS
from utils import utils
from utils.motion_model import motion_model

import pickle
import matplotlib.pyplot as plt

class Config:
    w_app = 0
    w_mot = 0.3
    w_app_inv = 0
    w_mot_inv = -0.3
    w_bbox = 0.7
    dth = 0.2
    initScore = 5

CONFIG = Config()

class MHT():
    """Multiply Hypothesis tracking class
    """

    def __init__(self, obj_num=0):
        """
        obj_id: start from 0, no bg
        """
        # get object number in this sequence
        self.obj_num = obj_num
        # Trees store all tracks
        self.trackTrees = {}
        for obj_id in range(self.obj_num):
            self.trackTrees[obj_id] = []
        # Nodes in the last frame (for gating)
        # { Tree Number:[node1,node2,...] }
        # self.currentNode = {}
        self.mwis = MWIS('track')
        # self.reid = ReidNetwork('best')
        # pre-process
        # get all re-id score
        # if os.path.exists(os.path.join(CONFIG.debug_path, 're_id', '%s.score'%sequence)):
        #     print('loading re-id score from disk -------------------')
        #     with open(os.path.join(CONFIG.debug_path, 're_id', '%s.score'%sequence),'rb') as f:
        #         self.reid_scores = pickle.load(f)
        # else:
        #     if not os.path.exists(os.path.join(CONFIG.debug_path, 're_id')):
        #         os.makedirs(os.path.join(CONFIG.debug_path, 're_id'))
        #     with open(os.path.join(CONFIG.debug_path, 're_id', '%s.score'%sequence),'wb') as f:
        #         self.reid_scores = self.reidAll()
        #         pickle.dump(self.reid_scores, f)
        # print(self.reid_scores[6])
        self.test_arr = []

    # def reidAll(self):
    #     """ make a list of the re-id score matrix
    #     format: [ array(M*N) ] * T, T means frame number, M means number of detection, N means number of obj
    #     """
    #     data = self.dataLoader.content
    #     reid_scores = []
    #     path, img_list = utils.get_obj_img(CONFIG, self.sequence)
    #     for detections in range(len(data)):
    #         print('processing frame %d ---------------'%detections)
    #         rois = data[detections]['rois'] # (N,4)
    #         roi_num = rois.shape[0]
    #         scores = np.zeros((roi_num, len(img_list)))
    #         path_target = os.path.join(CONFIG.img_dir, self.sequence, '%05d.jpg'%(detections))
    #         for roiId in range(roi_num):
    #             roi = rois[roiId] # (y1,x1,y2,x2)
    #             bbox = [roi[1], roi[0], roi[3], roi[2]]
    #             for i in range(len(img_list)):
    #                 scores[roiId, i] = self.reid.compute_score(path, path_target, img_list[i], bbox)
    #         reid_scores.append(scores)
    #     return reid_scores

    def iterTracking(self, data):
        # filter the data with detection score and overlap nms
        for i in range(len(data)): # len(data)
            print('current processing ------------------- '+str(i))
            # build and update track families
            self.formTrackFamily(data[i], i)
            # update the incompability list
            for treeId in range(len(self.trackTrees)):
                print('current object ------------------- '+str(treeId))
                # generate the global hypothesis
                paths, best_solution = self.treeToGraph(i+1, treeId)
                print('before pruning -------------------------------')
                for track in self.trackTrees[treeId]:
                    track.show()
                # N scan pruning
                self.nScanPruning(paths, best_solution, treeId)
                print('after pruning -------------------------------')
                for track in self.trackTrees[treeId]:
                    track.show()
                # save the output
                # tree.save2file('outs/tree/%d.txt'%i)
        # get best results
        results = []
        for obj_id in range(self.obj_num):
            results.append(self.findBestSolution(data, obj_id))
        return results


    def updateScore(self, scores, init=False):
        """Update score for each track
        inputs:
            Node: last node in tree (have previous score)
            appearance score (self.reid_scores)
            motion score (Mahalanobis distance and co-variance)
            scores = {'detection':scores[i], 'current_reid':current_reid, 'inverse_reid':inverse_reid,
                        'current_motion':current_motion, 'inverse_motion':inverse_motion, 'bbox_iou':bbox_iou}
        """
        # TODO: use iou to replace the distance score
        if init:
            # init score for root of a tree
            score = math.log(scores['detection'])
        else:
            # update by detection, distance and re-id score
            w_app = CONFIG.w_app
            w_mot = CONFIG.w_mot
            w_app_inv = CONFIG.w_app_inv
            w_mot_inv = CONFIG.w_mot_inv
            w_bbox = CONFIG.w_bbox
            # S_app = -1*math.log(0.5+0.5*math.exp(2.0*scores['current_reid'])) - math.log(CONFIG.c1)
            # S_app_inv = -1*math.log(0.5+0.5*math.exp(2.0*scores['inverse_reid'])) - math.log(CONFIG.c1)
            S_mot = scores['current_motion']
            S_mot_inv = scores['inverse_motion']
            S_mot_bbox = scores['bbox_iou']
            score = w_mot*S_mot + w_mot_inv*S_mot_inv + w_bbox*S_mot_bbox
            print('score -----------------------------------------')
            # print('S_app: %f'%S_app)
            # print('S_app_inv: %f'%S_app_inv)
            print('S_mot: %f'%S_mot)
            print('S_mot_inv: %f'%S_mot_inv)
            print('S_mot_bbox: %f'%S_mot_bbox)
            print('score: %f'%score)
            self.test_arr.append(score)
            print('-----------------------------------------------')
        return score


    def nScanPruning(self, paths, best_solution, treeNo, N=3, Bth=100):
        """Track Tree Pruning
        inputs:
            paths: list of track, {'treeId': treeId, 'track': [leaves to root], 'track_list': '0013424300'}
            best_solution: list of track number in path
            N: N-scan pruning approach
            Bth: branches number threshold
        """
        if paths == [] or best_solution == []:
            return
        T = len(paths[0]['track_list'])
        if T <= N:
            return
        # N Pruning
        ## Get k-(N-1) node in each tree, prun others
        for treeId in range(len(self.trackTrees[treeNo])):
            path_this_id = [paths[x] for x in best_solution if paths[x]['treeId']==treeId]
            node_in_path = []
            # find valid node in frame k-(N-1)
            for path in path_this_id:
                for track in path['track']:
                    if track.split('_')[1] == str(T-1-(N-1)):
                        node_in_path.append(track)
            # prune node if not in node_in_path
            node_in_tree = self.trackTrees[treeNo][treeId].filter_nodes(func=lambda x: x.identifier.split('_')[1] == str(T-1-(N-1)))
            node_names = [x.identifier for x in node_in_tree]
            for nodeId in node_names:
                # if nodeId.split('_')[1] == str(T-1-(N-1)):
                if nodeId not in node_in_path:
                    self.trackTrees[treeNo][treeId].remove_subtree(nodeId)
            '''
            # find K best and keep
            tracks = self.trackTrees[treeId].paths_to_leaves()
            if len(tracks) > Bth:
                # get weights and sort
                weights = [self.trackTrees[treeId].nodes[x[-1]].data['score'] for x in tracks]
                cut_index = 
            '''
        # remove empty tree
        self.trackTrees[treeNo] = [tree for tree in self.trackTrees[treeNo] if tree.nodes!={}]
        
    def findBestSolution(self, data, treeNo):
        paths, results = self.treeToGraph(len(data), treeNo, timeConflict=True)
        path_result = [paths[x] for x in results]
        # combine result in time
        roi_numbers = []
        for i in range(len(data)):
            track_list = []
            for path in path_result:
                track_list.append(path['track_list'][i])
            none_zeros = [x for x in track_list if x != 0]
            # none zero roi number should be 1 or many with the same value 
            if len(none_zeros) == 0:
                roi_numbers.append(-1)
            elif len(none_zeros) == 1:
                roi_numbers.append(none_zeros[0]-1)
            elif len(none_zeros) > 1:
                # first check the result
                if none_zeros[1:] == none_zeros[:-1]:
                    roi_numbers.append(none_zeros[0]-1)
                else:
                    raise Exception('same value in track, check wmis')
        return roi_numbers

    def treeToGraph(self, T, treeNo, timeConflict=False):
        """chnage all the tree in self.trackTrees in a Graph for MWIS.
        inputs:
            T: total time in the target video
        """
        return self.mwis.treeToGraph(T, self.trackTrees, treeNo, timeConflict)
        
            
    def generateGlobalHypothesis(self, T):
        pass

    def formTrackFamily(self, detections, t):
        """build track tree in self.trackTrees
        inputs:
            detections: the detection results in one frame {'rois':[N,4], 'scores':[N], 'class_ids':[N]}
            t: time, int, from 0, 0 is the ground truth
        """
        rois = detections['rois']
        detections_scores = detections['scores']
        class_ids = detections['class_ids']
        if t == 0:
            # build from ground truth
            for obj_id in range(self.obj_num):
                bbox = rois[obj_id]
                # create a root node
                updated_score = CONFIG.initScore # self.updateScore(1.0, 0, 0, 0, init=True)
                tree = Tree()
                tree.create_node(tag="T_"+str(t)+"_N_"+str(obj_id), identifier="t_"+str(t)+"_n_"+str(obj_id), 
                                    data={'score':updated_score, 'bbox':bbox, 'history_bbox':[bbox]})
                self.trackTrees[obj_id].append(tree)
        else:
            # Gating for node section    
            # Using Kalman Filter or TCN prediction
            '''
            tempCurrentNode = {}
            for treeID in self.currentNode:
                # for each node in self.currentNode, do gating process and add nodes
                for node in self.currentNode[treeID]:
                    self.
            '''
            tempCurrentNode = {}
            for obj_id in range(self.obj_num):
                nodeObjs = []
                for treeId in range(len(self.trackTrees[obj_id])):
                    for node in self.trackTrees[obj_id][treeId].leaves():
                        if int(node.identifier.split('_')[1]) == (t - 1):
                            # this node is from t-1 frame
                            #x_bar, P_bar = node.data['kf'].predict()
                            # predict bbox using history
                            # TODO: Find correct bbox
                            bbox_pred = motion_model(node.data['history_bbox'], t)
                            nodeObjs.append( {'treeId':treeId, 'node':node.identifier, 'obj_id': obj_id,
                                                'bbox_pred':bbox_pred, 'bbox':node.data['bbox']} )
                tempCurrentNode[obj_id] = nodeObjs
            print('roi number: %d'%rois.shape[0])
            # if the roi has no gate with any object
            for i in range(rois.shape[0]):
                print('current roi number is %d---------------------------------------------------'%i)
                roi_gating = False
                # for each detections, judge distance
                roi = rois[i] # (y1,x1,y2,x2)
                print(roi)
                bbox = [roi[1],roi[0],roi[3],roi[2]]

                ### new start
                roi_score = []
                for obj_id in range(self.obj_num):
                    obj_score = []
                    for nodeRecord in tempCurrentNode[obj_id]:
                        # compute for all of the score
                        
                        bbox_node = nodeRecord['bbox']
                        # compute scores
                        d = self.gating(bbox, bbox_node)
                        obj_score.append(d)
                    roi_score.append(obj_score)
                
                # we get roi bbox iou score with all nodes, now choose tree
                for obj_id in range(self.obj_num):
                    print('current obj is %d -----------------------'%obj_id)
                    # judge the app score of this roi
                    # current_reid = self.reid_scores[t][i,obj_id]
                    # print('re-id score is %f'%current_reid)
                    # if current_reid >= CONFIG.appScoreLimit:
                        # continue
                    # inverse_reid score is the min value of scores except current obj
                    # other_reids = [self.reid_scores[t][i,x] for x in range(len(self.reid_scores[t][i])) if x != obj_id]
                    # if other_reids == []:
                        # inverse_reid is None if there only one obj
                        # inverse_reid = 0
                    # else:
                        # inverse_reid = min(other_reids)
                    # get inverse motion score
                    other_motions = []
                    for j in range(len(roi_score)):
                        if j != obj_id:
                            other_motions.extend(roi_score[j])
                    if other_motions == []:
                        # inverse_reid is None if there only one obj
                        inverse_motion = 0
                    else:
                        inverse_motion = max(other_motions)
                    count = 0
                    for nodeId in range(len(tempCurrentNode[obj_id])):
                        print('current is roi %d, obj %d, node %s'%(i,obj_id,tempCurrentNode[obj_id][nodeId]['node']))
                        nodeRecord = tempCurrentNode[obj_id][nodeId]
                        # get current motion score
                        current_motion = roi_score[obj_id][nodeId]
                        # gating the node:
                        print('distance with last frame number %d is %f'\
                                %(int(nodeRecord['node'].split('_')[3]),current_motion))
                        if current_motion > CONFIG.dth:
                            # gating success 
                            # get bbox result
                            bbox_iou = utils.calc_bbox_iou(nodeRecord['bbox'], bbox)
                            # if no bbox iou, skip
                            if bbox_iou == 0:
                                continue
                            # get all the score we need for a new roi and target leaves
                            scores = {'detection':detections_scores[i],
                                        'current_motion':current_motion, 'inverse_motion':inverse_motion, 'bbox_iou':bbox_iou}
                            print(scores)
                            node_score = self.updateScore(scores)
                            current_score = self.trackTrees[obj_id][nodeRecord['treeId']]\
                                                            .nodes[nodeRecord['node']].data['score']
                            updated_score = node_score + current_score
                            # update history bbox
                            parent = self.trackTrees[obj_id][nodeRecord['treeId']]
                            history_bbox = self.trackTrees[obj_id][nodeRecord['treeId']]\
                                                    .nodes[nodeRecord['node']].data['history_bbox']
                            history_bbox.append(bbox)
                            # add node
                            print('creating node with %s'%(nodeRecord['node']))
                            self.trackTrees[obj_id][nodeRecord['treeId']]\
                                .create_node(tag="T_"+str(t)+"_N_"+str(i), 
                                    identifier="t_"+str(t)+"_n_"+str(i)+"_"+str(count), 
                                    parent=parent.nodes[nodeRecord['node']],
                                    data={'score':updated_score, 'bbox':bbox, 'history_bbox':history_bbox})
                            count = count + 1
                            roi_gating = True
                
                if not roi_gating:
                    # add a new tree to all tracks if the object not gate
                    for obj_id in range(self.obj_num):
                        # out of gating region
                        # add a new tree from the i-th detection
                        tree = Tree()
                        # create a root node
                        ## score
                        updated_score = self.updateScore({'detection':detections_scores[i]},init=True)
                        tree.create_node(tag="T_"+str(t)+"_N_"+str(i), identifier="t_"+str(t)+"_n_"+str(i), 
                                            data={'score':updated_score, 'bbox':bbox, 'history_bbox':[bbox]})
                        self.trackTrees[obj_id].append(tree)
                        


    def gating(self, curr_bbox, bbox):
        """Gating with predicted bbox
        curr_bbox: bbox of this roi
        bbox: [x1,y1,x2,y2] predicted
        """
        bbox_iou = utils.calc_bbox_iou(curr_bbox, bbox)
        return bbox_iou
    
    def update(self, out_boxes, out_scores, out_classes):
        data = [{
            'rois': out_boxes[i],
            'scores': out_scores[i],
            'class_ids': out_classes[i],
        } for i in range(len(out_scores))]
        self.iterTracking(data)

