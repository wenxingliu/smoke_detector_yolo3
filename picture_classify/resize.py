import os
import cv2 as cv

path =
all_file = os.listdir(path)
for img in all_file:
    cv.imread(path + img)
