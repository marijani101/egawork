# import  pandas as pd
# import json
#
# def read_and_append(node_path,relation_path):
#     nodes_file = pd.read_csv(node_path,skip_blank_lines=True)
#     relation_file = pd.read_csv(relation_path,skip_blank_lines=True)
#     print(nodes_file.head)
#     json_data = {}

# -*- coding:utf-8 -*-
import cv2
import numpy as np
from src.ocr import get_contours, stroke_width_transform, auto_color_correction
# import image
import os
import datetime
import matplotlib.pyplot as plt

# APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # project abs path
#
#
# target = os.path.join(APP_ROOT, 'static/')
# # savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "." + JPG
#
# destination = "/".join([target, "2019-10-18_14_40_04..JPG"])
image = cv2.imread('picha.jpg')
# image = cv2.resize(image,dsize=(800,1000))
# image = plt.imread("pics1.png")
# print(image)
# cv2.imshow('orig',image)
# cv2.waitKey(0)
image = auto_color_correction(image)
cv2.imshow('contoured', image)
cv2.waitKey(0)
# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', image)
cv2.waitKey(0)

# binary
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret2, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('second', thresh)
cv2.waitKey(0)
cv2.imwrite("result.jpg", thresh)
# gaussian blur to get images
# thresh = cv2.bilateralFilter(thresh,9,75,75)
# cv2.imshow('second',thresh)
# cv2.waitKey(0)

kernel_size = 11

blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 10  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 110  # minimum number of pixels making up a line
max_line_gap = 80  # maximum gap in pixels between connectable line segments
line_image = np.copy(image) * 0  # creating a blank to draw lines on
kernel = np.ones((11, 11), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)
# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
np.average(lines, axis=(1, 2))
a = 0
count = 0

for line in lines:
    for x1, y1, x2, y2 in line:
        a += (x2 - x1) / 2
        count += 1
av = a / count
print(av)
print(count)
for line in lines:
    for x1, y1, x2, y2 in line:
        avr = (x2 - x1) / 2
        if avr > av + 10:
            if avr > (edges.shape[1] / 4):
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 5)
        else:
            pass
lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
cv2.imwrite("yeay.jpg", lines_edges)

# dilation
kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)
cv2.imwrite("kernel.jpg", img_dilation)
# find contours
ctrs, im2 = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(ctrs)
# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
    # cv2.waitKey(0)

cv2.imshow('marked areas', image)

cv2.waitKey(0)
