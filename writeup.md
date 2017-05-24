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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Camera calibration

### 1. Using calibration images to calibrate camera

The camera calibration is done with the help of images given in this project. cv2.findChessboardCorners function is used to detect 9x6 interior chessboard corners in 20 images provided. This function is able to find 9x6 corners in 17 of 20 images as 3 other images do not have all 9x6 interior corners visible. Using these detected corners along with object points help cv2.calibrateCamera calibrate camera. The camera calibration parameters are then used to undistort a few images to ensure that image distortion can be rectified with calibrated camera. Following are a couple of images from chessboard corner detection in calibrating images.

### 2. Testing camera calibration with sample images

Below images verify that images can be undistorted with the help of camera calibration.

## Perspective transformation

### 1. Setting up perspective transformation for birds-eye view

A sample image (test_images/straight_lines1.jpg) is inspected in an image software to capture a trapezoid involving two lanes. The 4 vertices of these vertices are noted with their coordinates. Then, these coordinates are mapped to two parallel lines in birds-eye view so that the lanes in the image correspond to these two prominent  and parallel lines. cv2.getPerspectiveTransform function is used to create the matrix for this perspective transformation. Now this matrix can be used to transform an image into birds-eye view. The inverted matrix is also generated to faciliate inevrse transformation. 

### 2. Verifying perspective transformation

Another image (test_images/straight_lines2.jpg) is tested with the above perspective transformation to verify that its bird-eye view has two (almost) straight parallel lanes. Below are original image along with its perspective transformation.

## Lane finding pipeline

### 1. Using camera calibration to correct distortion in image

The parameters from above camera calibration are used to correct distortion in the image. Here are the original image and its undistorted version.

### 2. Generating thresholded binary image using combinations of color transforms and gradients

Initially, multiple sobel outputs (gradients in x and y directions, gradient magnitude and direction) are experimented to extract lanes in the binary image. Finally, thresholded sobel gradient along x direction and S component of HLS color space representation of the image are used to detect the lanes in thresholded binary image. Below is an output of thresholded binary image.

### 3. Applying perspective transformation to translate binary image in birds-eye view

The matrix for perspective transformation is applied on thresholded binary image to get its birds-eye view. Below is a transformed version of the binary image in birds-eye view.

### 4. Using histogram to identify left and right lanes and approximating the lanes with second-degree polynomials

Now, this step attempts to find strong signals around the left and right lanesand approximate these regions with some polynomials to model the lanes gemetrically. This step assumes that image thresholding and perspective transformation do a good job of isolating the left and right lanes and focusing on them in birds-eye view after perspective transformation.

First, bottom half of birds-eye view image is histogrammed along x-axis by counting non-zero pixels in y-axis. The strong peaks on left and right halves in this histogram give hints on where the lanes are located in this birds-eye view. The peak locations are noted and the image is further examined from bottom to top after starting from these peaks. A sliding window technique is utilized to traverse from bottom to top in order to gather non-zero pixels along the lanes. The image is divided into 9 horizontal strips and two windows (one for left lane and other for right lane) are slid from one strip to the next starting from bottom. Each sliding window starts at the center around peaks found in the histogram and collects its interior non-zero pixels. Then the window adjusts its center based on collected pixels if needed and moves to the next strip upwards. The sliding window process continues until windows hit top and by then, they have collected non-zero pixels along the lanes. Then finally groups of pixels contained in left and right sliding windows are fitted with second degree polynomial to approximate the lanes in birds eye view.

Below are a few pictures to depict the entire process.

### 5. Calculating the radii of lane curvatures and vehicle offset from center lane

The curvature radius can be found by using the curvature equation from second degree polynomial as show in the lecture notes. But those radii are applicable only in image. They need to be translated to world for proper measurement. So, first the polynomials are translated from pixel coordinates to world coordinates by simply using scaling polynomial coefficients with different factors. 

Let's say that following are scales to translate to world coordinates

    ym_per_pix = 30.0/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700   # meters per pixel in x dimension

So, if (X, Y) are the world coordinates that correspond to (x, y) in pixel space, then

    X = xm_per_pix * x
    Y = ym_per_pix * y

Now, x is fitted as polynomial of second degree of y.

    x = A * y * y + B * y + C

After replacing x and y in the above equation with world coordinates, we have the following.

    X = xm_per_pix / (ym_per_pix * ym_per_pix) * A * Y * Y + xm_per_pix / ym_per_pix * B * Y + xm_per_pix * C
      = A' * Y * Y + B' * Y + C'

where 

    A' = xm_per_pix / (ym_per_pix * ym_per_pix) * A,
    B' = xm_per_pix / ym_per_pix * B,  
    C' = xm_per_pix * C

Thus, polynomial coefficients can be scaled using the above scaling to translate to world coordinates. These new polynomial coefficients can be used to measure lane curvature radii in meters. These are also useful to measure drift of the car from the center of the lane.

### 6. Projecting the identified lanes in undistorted image

Finally, an image is constructed in birds-eye view to draw left and right lanes by following polynomial fits. This lane image is transformed back to original image space by doing inverse of perspective transformation. The transformed lane image is added to the original image to project lane detections.

## Video output

The above lane finding pipeline is applied to project video. As suggested in lecture notes, it's not necessary to find lanes from scratch in every frame. From one frame to next frame, locations of lanes change very little. So, lanes detected from previous frames can give useful guidence to detect lanes in current frame rather quickly. This optimization is adopted for detecting lanes in video. Basically, instead of histogramming binary image in birds-eye video to position windows for lane search and sliding windows from bottom to top to capture non-zero pixels for lanes, polynomials from previous frame are used to narrow down the binary image regions in order to gather these non-zero pixels along the left and right lanes. Specifically, bands of N pixels along y-axis are formed around polynomials from previous frame and non-zero pixels are captured in these bands in the current frame. These non-zero pixels are further fitted with second degree polynomials for the current frame to approximate the lanes in birds eye view.

## Files submitted
* Advanced_Lane_Lines.ipynb : ipython notebook for lane detection implementation
* output_images/result?.jpg : lane-annotated outputs for test images
* project_video_w_lanes.mp4 : project video output with projected lanes

## Discussions
