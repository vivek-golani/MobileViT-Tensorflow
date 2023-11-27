import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import skimage.measure as measure

from mobilevit.data.pipeline import load_label

def get_white_region_sizes(segmentation_map):
    white_pixels = segmentation_map == 255
    labeled_map = tf.py_function(lambda x: measure.label(x.numpy()), [white_pixels], tf.int32)
    segmentation_map = tf.cast(segmentation_map, tf.int32)
    num_regions = tf.reduce_max(labeled_map)

    max_region_label, max_region_size = 0, 0
    for region in range(1, num_regions + 1):
        region_mask = tf.equal(labeled_map, region)
        region_size = tf.reduce_sum(tf.cast(region_mask, tf.float32))
        if region_size > max_region_size:
            max_region_size = region_size
            max_region_label = region

    seg_map = tf.where(
        tf.equal(segmentation_map, 255), 
        tf.where(labeled_map == max_region_label, 255, 0),
        segmentation_map,
    )

    return seg_map

if __name__ == '__main__':
    pred_path = '/nfs/bigiris/vgolani/zebra/MobileViT-Tensorflow/visualizations/Train_same_set/0_pred_label.png'
    label_path = '/nfs/bigiris/vgolani/zebra/MobileViT-Tensorflow/visualizations/Train_same_set/0_label.png'
    vis_path = '/nfs/bigiris/vgolani/zebra/MobileViT-Tensorflow/visualizations/Train_same_set/0_pred_updated.png'

    seg_map = load_label(pred_path)
    seg_map_updated = get_white_region_sizes(seg_map)
    keras.utils.save_img(vis_path, seg_map_updated)



# def connected_components(pred_path, label_path):
    
#     label = load_label(label_path)

#     ROWS, COLS, _ = seg_map.shape
#     visit = {}

#     for r in range(ROWS):
#         for c in range(COLS):
#             visit[(r,c)] = 0

#     def dfs(r, c, vis):
#         if(r<0 or r>=ROWS and c<0 and c>=COLS or seg_map[r][c] == 0):
#             return 0
        
#         visit[(r,c)] = vis
#         area = 0
#         for dr in [0,0,-1,-1,-1,1,1,1]:
#             for dc in [-1,1,-1,0,1,-1,0,1]:
#                 if ((r,c) not in visit.keys()):
#                     area += 1+dfs(r+dr, c+dc, vis)
        
#         return area        

#     vis = 0
#     keep, max_area = 0, 0

#     for r in range(ROWS):
#         for c in range(COLS):
#             if seg_map[r][c] == 255:
#                 vis += 1
#                 area = dfs(r, c, vis)
#                 if area > max_area:
#                     max_area = area
#                     keep = vis 

#     for r in range(ROWS):
#         for c in  range(COLS):
#             if visit[(r, c)] != keep:
#                 seg_map[r][c]= 0

#     return seg_map