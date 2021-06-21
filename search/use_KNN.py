import json
from build_knn.KNN_model import Jugde
import tensorflow as tf
# from api.v1.core.config import KNN_THRESHOLD, KNN_TRUE_LABEL
# from api.logger import logger_mess

# logger = logger_mess('use KNN')

KNN_THRESHOLD, KNN_TRUE_LABEL = 0.5, 2


KNN_model = Jugde("./search/KNN_data/knn_searcher.bin")
# latest = tf.train.latest_checkpoint("./embedding_model")


def load_label_dict(label_dict_path="./search/KNN_data/label_dict.json"):
    json_file = open(label_dict_path, "r", encoding='utf-8')
    label_dict = json.load(json_file)
    return label_dict


def save_label_dict(label_dict, label_dict_path="./search/KNN_data/label_dict.json"):
    new_json_file = open(label_dict_path, "w")
    json.dump(label_dict, new_json_file, indent=4)


def evaluate_sample(embedding_vector, num_res):
    label_dict = load_label_dict()
    num_res = num_res if num_res < len(label_dict) else len(label_dict)
    predict, distance = KNN_model.find_item(embedding_vector, num_res)
    # logger.info('Distance' + str(distance))
    ids = []
    for i, d in enumerate(distance[0]):
        if float(d) > KNN_THRESHOLD:
            continue
        try:
            index = list(label_dict.keys())[list(label_dict.values()).index(int(predict[0][i]))]
            print('Get label name for distance: ' + str(d) + '\t' + str(index))
            ids.append([index, float(d)])
        except:
            # logger.error('Something wrong')
            raise ValueError
    # return ids
    return check_distance(ids) if ids else []


def check_distance(sake_ids):
    dis_label = sake_ids[0][1]
    if dis_label <= KNN_TRUE_LABEL:
        return [sake[0] for sake in sake_ids]
    else:
        return []
