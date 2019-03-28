# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:04:45 2019

@author: khira
"""

import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True,
#	help = "path to the image file")
#args = vars(ap.parse_args())

args = {"image":'G:\\Desktop\\AI_ML\\Product detection\\barcodeImages\\300x0w.jpg'}
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
cv2.waitKey(0)
 
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
 
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

cv2.imshow("Image", gradient)
cv2.waitKey(0)

blurred = cv2.blur(gradient, (9, 9))

# blurred = cv2.blur(gray, (9, 9))
cv2.imshow("Image", blurred)
cv2.waitKey(0)

(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

cv2.imshow("Image", thresh)
cv2.waitKey(0)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Image", closed)
cv2.waitKey(0)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 2)
cv2.imshow("Image", closed)
cv2.waitKey(0)

closed = cv2.dilate(closed, None, iterations = 4)
cv2.imshow("Image", closed)
cv2.waitKey(0)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
 
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)