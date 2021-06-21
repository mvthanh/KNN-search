import os
import cv2
import numpy as np
from model_ebd.load_model import load_facenet
from search.use_KNN import evaluate_sample

face_net = load_facenet()


def get_img_ebd(img_path):
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (160, 160))

    face_pixels = img_cv.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    img_np = np.asarray([face_pixels])
    return face_net.predict(img_np)[0]


if __name__ == '__main__':
    image_path = r'D:\BAP\data_BAP9\vinhna\14.png'
    ebd_vector = get_img_ebd(image_path)
    print(evaluate_sample(ebd_vector, 7))
