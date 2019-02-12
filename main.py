

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread("ive.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 0)
print('here')
print(rects)

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    print("shape")
    print(shape)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (255, 255, 0), -1)
        # cv2.circle(image, center, radian, color, thickness) 1 이면 원 안쪽을 채움

cv2.imshow("Output", image)
cv2.waitKey(0)

