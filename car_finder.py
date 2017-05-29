from lessons_functions import *
from scipy.ndimage.measurements import label
import time
import numpy as np
from collections import deque

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

class CarFinder:
    def __init__(self, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, hog_channel, spatial_feat, hist_feat, hog_feat):
      self.found_cars = 0
      self.svc = svc
      self.X_scaler = X_scaler
      self.orient = orient
      self.pix_per_cell = pix_per_cell
      self.cell_per_block = cell_per_block
      self.spatial_size = spatial_size
      self.hist_bins = hist_bins
      self.color_space = color_space
      self.windows = None
      self.image_sizes = [310, 256, 128, 96, 64, 32]
      self.bottom_thresholds = [0, 30, 50, 75, 100, 150]
      self.threshold_top = 30
      self.threshold_left = 30
      self.threshold_right = 30
      self.threshold_bottom = 0
      self.x_start_stop = [0, 0]
      self.y_start_stop = [0, 0]
      self.scale = 1.5
      self.heat = None
      self.hog_channel = hog_channel
      self.spatial_feat = spatial_feat
      self.hist_feat = hist_feat
      self.hog_feat = hog_feat
      self.frames = 0
      self.draw_img = None
      self.history = deque(maxlen = 10)

    def output_info(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Found Cars = %d' % self.found_cars, (50, 150), font, 1, (255, 0, 0), 2)
        return img

    def process(self, img):
      sample = 6
      if self.windows is None:
        self.windows = multiple_slide_windows(img, self.image_sizes, self.threshold_left, self.threshold_right, self.threshold_top, self.bottom_thresholds, xy_overlap=(0.8, 0.8))
  
      if self.draw_img is None or self.frames % sample is 0:   
        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        hot_windows = search_windows(img, self.windows, self.svc, self.X_scaler, color_space=self.color_space, 
                          spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                          orient=self.orient, pix_per_cell=self.pix_per_cell, 
                          cell_per_block=self.cell_per_block, 
                          hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                          hist_feat=self.hist_feat, hog_feat=self.hog_feat) 
        heat = add_heat(heat, hot_windows)
      else:
        heat = self.heat

      bboxes = find_cars(img, ystart=400, ystop=656, scale=self.scale, svc=self.svc, X_scaler=self.X_scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)
      heat = add_heat(heat, bboxes)

      # Apply threshold to help remove false positives
      heat = apply_threshold(heat, 2)

      # Visualize the heatmap when displaying    
      heatmap = np.clip(heat, 0, 255)

      # Find final boxes from heatmap using label function
      labels = label(heatmap)
      
      self.history.append(heat)
      accumulated_heat = sum(self.history)
      accumulated_heat = apply_threshold(accumulated_heat, 6)
      accumulated_heatmap = np.clip(accumulated_heat, 0, 255)
      accumulated_labels = label(accumulated_heatmap)
      
      self.draw_img = draw_labeled_bboxes(np.copy(img), accumulated_labels)

      self.found_cars = accumulated_labels[1]

      self.heat = heat

      self.frames += 1
      return self.draw_img