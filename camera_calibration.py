import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os.path

def get_calibration(camera_calibration_pickle = 'dist_pickle.p', force=False):
  if force is True or not os.path.isfile(camera_calibration_pickle):
    return calibrate()
  else:
    return load_calibrate()

def calibrate(inner_chessboard_corners = (9,6), camera_calibration_images = 'camera_cal/calibra*.jpg', camera_calibration_pickle = 'dist_pickle.p', force=False):

  objp = np.zeros((inner_chessboard_corners[1]*inner_chessboard_corners[0],3), np.float32)
  objp[:,:2] = np.mgrid[0:inner_chessboard_corners[0], 0:inner_chessboard_corners[1]].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d points in real world space
  imgpoints = [] # 2d points in image plane.

  # Make a list of calibration images
  images = glob.glob(camera_calibration_images)

  img_size = None
  for idx, fname in enumerate(images):
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      ret, corners = cv2.findChessboardCorners(gray, inner_chessboard_corners, None)

      if img_size is None:
        img_size = (img.shape[1], img.shape[0])

      if ret == True:
          objpoints.append(objp)
          imgpoints.append(corners)

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  dist_pickle["objpoints"] = objpoints
  dist_pickle["imgpoints"] = imgpoints

  pickle.dump( dist_pickle, open(camera_calibration_pickle, "wb" ) )

  return mtx, dist, objpoints, imgpoints

def load_calibrate(camera_calibration_pickle = 'dist_pickle.p'):
  dist_pickle = pickle.load( open(camera_calibration_pickle, "rb" ) )
  mtx = dist_pickle["mtx"]
  dist = dist_pickle["dist"]
  objpoints = dist_pickle["objpoints"]
  imgpoints = dist_pickle["imgpoints"]

  return mtx, dist, objpoints, imgpoints