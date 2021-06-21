import numpy as np
import cv2
from copy import deepcopy
import os
from tqdm import tqdm


def aug_light(img, imgfile, out_dir):
    gama = [0.8, 0.9, 1.1, 1.2]
    for g in gama:
        img_agu = deepcopy(img)
        img_agu = img_agu / 255
        img_agu = (img_agu ** g) * 255
        img_agu = img_agu.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, imgfile.replace('.', f'-{g}.')), img_agu)


def rand_file(img_dir):
    files = os.listdir(img_dir)
    n = len(files)
    rand = np.random.permutation(n)
    return [files[int(r)] for r in rand[:300]]


def main():
    in_dir = r'D:\BAP\Data_out_t9'

    for dir in tqdm(os.listdir(in_dir)):
        img_dir = os.path.join(in_dir, dir)

        for file in rand_file(img_dir):
            img = cv2.imread(os.path.join(img_dir, file))
            aug_light(img, file, img_dir)


if __name__ == '__main__':
    main()

    exit(0)

    path = r'C:\Users\TechCare\Desktop\KNN\KNN-search\resources\avatar.jpg'
    out_dir = r'C:\Users\TechCare\Desktop\KNN\Data'
    # img BRG
    image = cv2.imread(path)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    gama = .85
    img = image/255
    img = (img**gama) * 255
    img = img.astype(np.uint8)
    cv2.imshow('img2', img)
    cv2.waitKey(0)
