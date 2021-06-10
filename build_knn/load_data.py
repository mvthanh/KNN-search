import os
import cv2
import numpy as np
import json


def load_label_dict(label_dict_path="./label_dict.json"):
    json_file = open(label_dict_path, "r", encoding='utf-8')
    label_dict = json.load(json_file)
    return label_dict


def save_label_dict(label_dict, label_dict_path="./label_dict.json"):
    new_json_file = open(label_dict_path, "w")
    json.dump(label_dict, new_json_file, indent=4)


def load_data(file_path):
    '''
    This function load data from a input file directory, then output train data and their label as numpy arrays.

    Input:
        - file_path (str): Directory to the training data.
        The directory should be organized as follow:

            main_directory/
                ...class_a/
                ......a_image_1.jpg
                ......a_image_2.jpg
                ...class_b/
                ......b_image_1.jpg
                ......b_image_2.jpg
    Output:
        - X (numpy array): RGB information of images as numpy array
        - y (numpy array): Label of images as numpy array
    '''

    # Load label dictionary
    label_dict = load_label_dict()
    # Load sub-directories
    dirs = os.listdir(file_path)
    X = []
    y = []
    for i, dir_ in enumerate(sorted(dirs)):
        # If label (in this case is dir_) is not from the dictionary, we add that label to the dictionary
        if (dir_ not in label_dict):
            label_dict[dir_] = len(label_dict)
        # Read the image dictionary
        data_dir = os.path.join(file_path, dir_)
        list_imgs = os.listdir(data_dir)
        list_img_cv = []
        type_list = []
        for img_file in list_imgs:
            abs_dir = os.path.join(data_dir, img_file)
            img_cv = cv2.imread(abs_dir, 1)
            img_cv = cv2.resize(img_cv, (224, 224))
            X.append(img_cv)
            y.append(label_dict[dir_])
    save_label_dict(label_dict)
    X = np.array(X)
    # print(X.shape)
    X = np.squeeze(X)
    y = np.asarray(y)
    return X, y
