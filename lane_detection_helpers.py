from scipy.signal import medfilt
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        
    abs_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(abs_sobel)
    binary_output[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output



# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_thresh(img, s_thresh=(200, 255), sx_thresh=(50, 100)):
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    return color_binary

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    bin = np.zeros_like(S)
    bin[(S > thresh[0]) & (S <= thresh[1])] = 1
    return bin

def bin_to_rgb(bin_image):
    return cv2.cvtColor(bin_image*255, cv2.COLOR_GRAY2RGB)

def hls_to_rgb(hlsimage):
    return cv2.cvtColor(hlsimage, cv2.COLOR_HLS2RGB)

def binary_noise_reduction(img, thresh=4):      
    k = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img

def create_binary(img):
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply each of the thresholding functions
    gradx = binary_noise_reduction(abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=threshold))
    grady = binary_noise_reduction(abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=threshold))
    mag_binary = binary_noise_reduction(mag_thresh(img, sobel_kernel=mag_sobel_kernel, mag_thresh=mag_threshold))
    dir_binary = binary_noise_reduction(dir_threshold(img, sobel_kernel=binary_sobel_kernel, thresh=binary_mag_threshold))
    color_binary = binary_noise_reduction(color_thresh(img, s_thresh=s_thresh, sx_thresh=sx_thresh))
    
    return (gradx, grady, mag_binary, dir_binary, color_binary)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_steering_correction(steering_angle = 0, max_length = 100, boost = 2):
    length = (steering_angle * boost * max_length) / 180
    if (length > max_length):
        length = max_length
    return length

def get_trapezoid(img, steering_angle = 0, boost = 2, horizon = 0, margin = 0):
    ysize = img.shape[0]
    xsize = img.shape[1]
    left_correction=0
    right_correction=0
    
    if (steering_angle < 0):
        left_correction = get_steering_correction(steering_angle, round(xsize/2), boost)
    else:
        right_correction = get_steering_correction(steering_angle, round(xsize/2), boost)
        
    return  np.array([[
                (round(xsize/2)+left_correction, round(ysize*horizon/100)),
                (round(xsize/2)+right_correction, round(ysize*horizon/100)),
                (xsize - margin, ysize),
                (0 + margin, ysize),
            ]], dtype=np.int32)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

def histogram_peaks(image):
    histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)
    x_margins = (int(image.shape[1]*0.08), int(image.shape[1]*0.92))
    
    idx = [i for i in range(histogram.shape[0]) if i < x_margins[0] or i > x_margins[1]]
    histogram[idx] = 0    
    histogram = medfilt(histogram, 7)
    
    midpoint = np.int(histogram.shape[0]/2)
    left_base_x = np.argmax(histogram[:midpoint])
    right_base_x = np.argmax(histogram[midpoint:]) + midpoint

    return left_base_x, right_base_x, midpoint

def find_sliding_window(binary_warped):
    leftx_base, rightx_base, midpoint = histogram_peaks(binary_warped)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low) & 
            (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) & 
            (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & 
            (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) & 
            (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return out_img, left_fit, left_fitx, right_fit, right_fitx, ploty

def skip_sliding_window(binary_warped, left_fit, right_fit, margin = 125, quadratic_coeff = 3e-4):
    leftx_base, rightx_base, midpoint = histogram_peaks(binary_warped)
    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return out_img, left_fit, left_fitx, right_fit, right_fitx, ploty

def curve_rad(fit, y):
    return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])

def get_curve_rad(ploty, leftx, rightx, y_eval, xm_per_pix = 3.7/700, ym_per_pix = 21/720):
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_curverad, right_curverad)

def get_lane_deviation(img, xm_per_pix = 3.7/700):
    center = img.shape[0] / 2
    lane_deviation = (center - img.shape[1] / 2.0) * xm_per_pix
    return lane_deviation

def pipeline(img, steering_angle = 0, horizon = 52):
    thresh_s = (120, 255)
    thresh_l = (40, 255)
    thresh_sobel = (20, 255)
    
    # thresh_mag = (0, 255)
    # thresh_dir = (0, 255)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
    l_channel = hls[:,:,1]       
    s_channel = hls[:,:,2]

    # Gradient, Magnitude, Direction Thresholds
    gradx = abs_sobel_thresh(l_channel, orient='x', thresh=thresh_sobel)
    # mag_bin = mag_thresh(img, sobel_kernel=ksize, mag_thresh=thresh_mag)
    # dir_bin = dir_threshold(img, sobel_kernel=ksize, thresh=thresh_dir)
    
    # Threshold l (lightness) channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1])] = 1
    
    # Threshold s (saturation) channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1
    
    # Combined Binary
    combined = np.zeros_like(gradx)
    combined[(((l_binary == 1) & (s_binary == 1)) | (gradx == 1))] = 1
    
    unnoised = binary_noise_reduction(combined)
    
    kernel_size = 1 # kernel size must be odd
    gaussian_blur_image = gaussian_blur(unnoised, kernel_size)
    
    interesting_region = get_trapezoid(gaussian_blur_image, steering_angle=steering_angle, horizon=horizon, margin = 150)
    interesting_region_image = region_of_interest(gaussian_blur_image, interesting_region)
    
    return interesting_region_image