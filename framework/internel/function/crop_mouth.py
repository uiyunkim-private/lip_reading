
from imutils import face_utils
import cv2
import numpy as np


def crop_mouth( image,face_shape,face_detector,face_predictor):
    rects = face_detector(image, 0)
    if len(rects) != 1:
        return None

    shape = face_predictor(image, rects[0])
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
    ratio = 70 / w

    image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
    x = x * ratio
    y = y * ratio
    w = w * ratio
    h = h * ratio
    midy = y + h / 2
    midx = x + w / 2
    xshape = face_shape[1] / 2
    yshape = face_shape[0] / 2

    mouth_image = image[int(midy - yshape):int(midy + yshape), int(midx - xshape):int(midx + xshape)]
    return mouth_image