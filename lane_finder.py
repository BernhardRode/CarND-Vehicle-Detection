from scipy.signal import medfilt
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from lane_detection_helpers import *

class LaneFinder:
    def __init__(self, mtx, dist):
        self.frames = 0
        self.mtx = mtx
        self.dist = dist
        
        self.ym_per_pix = 21/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.quadratic_coeff = 3e-4
        
        self.img = None
        self.undist = None
        self.binary = None
        self.binary_warped = None
        self.output = None
        
        self.avg_curverad = None
        self.left_curverad = None
        self.right_curverad = None
        self.offset = None
        
        self.fit_cr = None
        self.left_fit = None
        self.left_fitx = None 
        self.right_fit = None 
        self.right_fitx = None 
        self.ploty = None
        self.y_eval = None
        self.points = None
        
        self.center = None
        
        self.src = None
        self.dst = None
        
        self.M = None
        self.Minv = None
        
        self.buffer = 50
        
    def output_info(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Curve rad. = %d(m)' % self.avg_curverad, (50, 50), font, 1, (255, 0, 0), 2)
        cv2.putText(img, 'Center Deviation = %.2fm' % self.offset, (50, 100), font, 1, (255, 0, 0), 2)
        return img
        
    def generate_output(self):
        warp_zero = np.zeros(self.undist.shape[0:2], np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        cv2.fillPoly(color_warp, np.int_([self.points]), (0, 255, 0))
        
        img_size = (self.img.shape[1], self.img.shape[0])
        unwarped = warp(color_warp, self.Minv)
        result = cv2.addWeighted(self.undist, 1, unwarped, 0.3, 0)
        info = self.output_info(result)
        
        self.output = info
        
    def update(self, left_fit, left_fitx, right_fit, right_fitx, ploty):    
        points_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((points_left, points_right))

        if self.buffer < 50:
            a = 0.8 ** self.buffer
            self.points = a * self.points + (1-a) * points
            self.left_fit = a * self.left_fit + (1-a) * left_fit
            self.right_fit = a * self.right_fit + (1-a) * right_fit
        else:
            self.points = points
            self.left_fit = left_fit
            self.right_fit = right_fit  

        self.left_fitx = left_fitx 
        self.right_fitx = right_fitx 
        self.ploty = ploty

        y_eval = np.max(ploty)
        left_curverad, right_curverad = get_curve_rad(self.ploty, self.left_fitx, self.right_fitx, y_eval)

        self.left_curverad = left_curverad
        self.right_curverad = right_curverad
        self.avg_curverad = (left_curverad + right_curverad) / 2

        self.center = (left_fitx[-1] + right_fitx[-1]) / 2.0
        self.offset = ((self.img.shape[1]) / 2.0 - self.center) * self.xm_per_pix

    def process(self, img):
        self.img = np.copy(img)
        
        if self.src is None and self.dst is None:
            self.src = np.float32([
                (200, 720),
                (587, 455),
                (696, 455),
                (1105, 720)])

            self.dst = np.float32([
                (self.src[0][0] + 150, 720),
                (self.src[0][0] + 150, 0),
                (self.src[-1][0] - 150, 0),
                (self.src[-1][0] - 150, 720)])
            
            self.M = cv2.getPerspectiveTransform(self.src, self.dst)
            self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
            
        self.undist = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)
        self.binary = pipeline(self.undist)
        self.binary_warped = warp(self.binary, self.M)
        
        try:
            if self.left_fit is None and self.right_fit is None:
                raise Exception('No sliding window')  
            out_img, left_fit, left_fitx, right_fit, right_fitx, ploty = skip_sliding_window(self.binary_warped, self.left_fit, self.right_fit)
            
            threshold = 100
            if (self.center < np.average(left_fitx) + threshold or self.center > np.average(right_fitx) - threshold):
                raise Exception('Sliding window failed, rerun and get correct values.')  
            
        except Exception as e:
            # When exception, fallback to sliding window
            try:
                out_img, left_fit, left_fitx, right_fit, right_fitx, ploty = find_sliding_window(self.binary_warped)
            except Exception as e:
                if self.output is not None:
                    return self.output
                else:
                    return self.undist
        self.out_img = out_img
        
        # print(right_fitx)
        
        self.update(left_fit, left_fitx, right_fit, right_fitx, ploty)
        self.frames = self.frames + 1
        self.generate_output()
        
        return self.output