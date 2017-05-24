# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_corners1.png
[image2]: ./output_images/chessboard_corners2.png
[image3]: ./output_images/camera_undistortion_example.png
[image4]: ./output_images/perspective_correction_example.png
[image5]: ./output_images/warped_binary.png
[image6]: ./output_images/histogram.png
[image7]: ./output_images/sliding_window.png
[image8]: ./output_images/poly_fit_lanes.png
[image9]: ./output_images/result_lanes.png

## Camera calibration

### 1. Using calibration images to calibrate camera

The camera calibration is done with the help of images given in this project. cv2.findChessboardCorners function is used to detect 9x6 interior chessboard corners in 20 images provided. This function is able to find 9x6 corners in 17 of 20 images as 3 other images do not have all 9x6 interior corners visible. Using these detected corners along with object points help cv2.calibrateCamera calibrate camera. The camera calibration parameters are then used to undistort a few images to ensure that image distortion can be rectified with calibration matrix and distortion coefficients. Following are a couple of images from chessboard corner detection in calibrating images.

![alt text][image1]
![alt text][image2]

### 2. Testing camera calibration with sample images

Below images verify that images can be undistorted with the help of camera calibration.

![alt text][image3]

## Perspective transformation

### 1. Setting up perspective transformation for birds-eye view

A sample image (test_images/straight_lines1.jpg) is inspected in an image software to capture a trapezoid involving two straight line lanes. The 4 vertices of this trapezoid are noted with their coordinates. Then, these coordinates are mapped to two parallel lines in birds-eye view so that the straight lanes in the image correspond to these two prominent and parallel lines. cv2.getPerspectiveTransform function is used to create the matrix for this perspective transformation. Now this matrix can be used to transform an image into birds-eye view. The inverted matrix is also generated to faciliate inverse transformation. 

### 2. Verifying perspective transformation

Another image (test_images/straight_lines2.jpg) is tested with the above perspective transformation to verify that its bird-eye view has two (almost) straight parallel lanes. Below are original image along with its perspective transformation into birds-eye view.

![alt text][image4]

## Lane finding pipeline

### 1. Using camera calibration to correct distortion in image

The parameters from above camera calibration are used to correct distortion in the image.

### 2. Generating thresholded binary image using combinations of color transforms and gradients

Initially, multiple sobel outputs (gradients in x and y directions, gradient magnitude and direction) are experimented to extract lanes in the binary image. After a lot of experimentations, thresholded sobel gradient along x direction and S component of HLS color space representation of the image are used to detect the lanes in thresholded binary image.

### 3. Applying perspective transformation to translate binary image in birds-eye view

The matrix for perspective transformation is applied on thresholded binary image to get its birds-eye view. Below are original image, undistorted image, thresholded binary image and a transformed version of the binary image in birds-eye view.

![alt text][image5]

### 4. Using histogram to identify left and right lanes and approximating the lanes with second-degree polynomials

Now, this step attempts to find strong signals around the left and right lanes and approximate these regions with some polynomials to model the lanes geometrically. This step assumes that image thresholding and perspective transformation do a good job of isolating the left and right lanes and focusing on them in birds-eye view after perspective transformation.

First, bottom half of birds-eye view image is histogrammed along x-axis by counting non-zero pixels in y-axis. The strong peaks on left and right halves in this histogram give hints on where the lanes are located in this birds-eye view. The peak locations are noted and the image is further examined from bottom to top after starting from these peaks at the bottom. A sliding window technique is utilized to traverse from bottom to top in order to gather non-zero pixels along the lanes. The image is divided into 9 horizontal strips and two windows (one for left lane and other for right lane) are slid from one strip to the next starting from bottom. Each sliding window starts at the center around peaks found in the histogram and collects its interior non-zero pixels. Then the window adjusts its center based on collected pixels if needed and moves to the next strip upwards. The sliding window process continues until windows hit top and by then, they have collected non-zero pixels along the lanes. Then finally, groups of pixels contained in left and right sliding windows are fitted with second degree polynomial to approximate the lanes in birds eye view.

Below are a few pictures to depict the entire process.

![alt text][image6]

![alt text][image7]

![alt text][image8]

### 5. Calculating the radii of lane curvatures and vehicle offset from center lane

The curvature radius can be found by using the curvature equation from second degree polynomial as shown in the lecture notes. But those radii are applicable only in image pixel coordinates. They need to be translated to world coordinates for proper measurement. So, first the polynomials are translated from pixel coordinates to world coordinates by simply using scaling polynomial coefficients with different factors. 

Let's say that following are scales to translate to world coordinates.

    ym_per_pix = 30.0/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700   # meters per pixel in x dimension

So, if (X, Y) are the world coordinates that correspond to (x, y) in pixel coordinate space, then

    X = xm_per_pix * x
    Y = ym_per_pix * y

Now, x is fitted as polynomial of second degree of y from the above lane finding technique.

    x = A * y * y + B * y + C

After replacing x and y in the above equation with world coordinates, we have the following.

    X = xm_per_pix / (ym_per_pix * ym_per_pix) * A * Y * Y + xm_per_pix / ym_per_pix * B * Y + xm_per_pix * C
      = A' * Y * Y + B' * Y + C'

where 

    A' = xm_per_pix / (ym_per_pix * ym_per_pix) * A,
    B' = xm_per_pix / ym_per_pix * B,  
    C' = xm_per_pix * C

Thus, polynomial coefficients can be scaled using the above scaling to translate to world coordinate space. These new polynomial coefficients can be used to measure lane curvature radii in meters. These are also useful to measure drift of the car from the center of the lane.

### 6. Projecting the identified lanes in undistorted image

Finally, an image is constructed in birds-eye view to draw left and right lanes by following polynomial fits. This lane image is transformed back to original image space by doing inverse of perspective transformation. The transformed lane image is added to the original image to project lane detections.

![alt text][image9]

## Video output

The above lane finding pipeline is applied to project video. As suggested in lecture notes, it's not necessary to find lanes from scratch in every frame. From one frame to next frame, locations of lanes change very little. So, lanes detected from previous frames can give useful guidence to detect lanes in current frame rather quickly. This optimization is adopted for detecting lanes in video. Basically, instead of histogramming binary image in birds-eye video to position windows for lane search and sliding windows from bottom to top to capture non-zero pixels for lanes, polynomials from previous frame are used to narrow down the binary image regions in order to gather these non-zero pixels along the left and right lanes. Specifically, bands of N pixels along y-axis are formed around polynomials from previous frame and non-zero pixels are captured in these bands in the current frame. These non-zero pixels are further fitted with second degree polynomials for the current frame to approximate the lanes in birds eye view.

## Files submitted
* Advanced_Lane_Lines.ipynb : ipython notebook for lane detection implementation
* output_images/result?.jpg : lane-annotated outputs for test images
* project_video_w_lanes.mp4 : project video output with projected lanes

## Discussions

Tweaking gradient thresholds and color space thresholds was challenging in order to identify left and right lanes in test images and project video under different conditions. Current selection of thresholding on sobel gradient along x and S channel of HLS color space worked fine with given images and project video. But the current thresholds may not work under different road conditions and under different light conditions. For unpaved road conditions and for dark lighting conditions, H channel of color space may be useful. Also taking gradient direction into account could be helpful to make lane detection more robust.

Perspective transformation may cause a bunch of unwanted non-zero pixels from region of interest to concentrate in warped binary image. This could cause unwanted areas to become peaks in histogram and lanes could be misdetected. This particularly can happen in very curvy areas with detectable objects like trees, road separators near the lane. Reducing the projection in y direction in perspective setup would help reduce this kind of issue.

Finally, the polynomial fits for left and right lanes can be compared with eath other to make lane detection more robust. If one polynomial is way off the other, more can be examined to generate higher confidence on one of the two polynomials and polynomial for other low confidence lane can be rectified with the help of polynomial for high confidence lane.
