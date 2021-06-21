from build_knn.load_data import *


def train_KNN_folder(file_path, embedding_model, KNN_model):
    """
    This function is used for training KNN for a whole folder

    Input:
        file_path (str): Training directory
        embedding_model (keras_model): Keras model for embedding
        KNN_model (Jugde): KNN model for querying
    Output:
        KNN_model (Jugde): KNN model after add elements and training
    """
    dirs = os.listdir(file_path)
    X = []
    y = []
    label_dict = load_label_dict()
    add_new = False
    for i, dir_ in enumerate(sorted(dirs)):
        print("Input ", dir_)
        if dir_ not in label_dict:
            label_dict[dir_] = len(label_dict)
            add_new = True
        data_dir = os.path.join(file_path, dir_)
        list_imgs = os.listdir(data_dir)
        list_img_cv = []
        type_list = []
        for img_file in list_imgs:
            abs_dir = os.path.join(data_dir, img_file)
            img_cv = cv2.imread(abs_dir)
            img_cv = cv2.resize(img_cv, (160, 160))

            face_pixels = img_cv.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std

            img_np = np.asarray([face_pixels])
            X.append(img_np)
            y.append(label_dict[dir_])
    if add_new:
        save_label_dict(label_dict)
    X = np.asarray(X)
    X = np.squeeze(X)
    X = embedding_model.predict(X)
    KNN_model.add_items((X, y))
    KNN_model.save_KNN()

    return KNN_model


def train_KNN_one(file_path, label, embedding_model, KNN_model):
    '''
    This function is used for training KNN for one data point

    Input:
        file_path (str): Training point directory
        label (str): Label of the data point (example: Asus, Apple,...)
        embedding_model (keras_model): Keras model for embedding
        KNN_model (Jugde): KNN model for querying
    Output:
        KNN_model (Jugde): KNN model after add elements and training
    '''
    label_dict = load_label_dict()
    add_new = False
    if label not in label_dict:
        label_dict[label] = len(label_dict)
        add_new = True
    img_cv = cv2.imread(file_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    img_np = np.asarray([img_cv])
    embedding_vector = embedding_model.predict(img_np)
    KNN_model.add_items((embedding_vector, label_dict[label]))
    if add_new:
        save_label_dict(label_dict)
    KNN_model.save_KNN()
    return KNN_model


def evaluate_folder(file_path, embedding_model, KNN_model):
    '''
    This function is used to evaluate a whole folder.
    Or we can call it the testing phase

    Input:
        file_path (str): Testing directory
        embedding_model (keras_model): Keras model for embedding
        KNN_model (Jugde): KNN model for querying
    Output:
        Accuracy of the models
    '''
    count = 0
    correct = 0
    dirs = os.listdir(file_path)
    label_dict = load_label_dict()
    X = []
    y = []
    for i, dir_ in enumerate(sorted(dirs)):
        print("Input ", dir_)
        if (dir_ not in label_dict):
            print(dir_, "is not exists ")
            continue
        data_dir = os.path.join(file_path, dir_)
        list_imgs = os.listdir(data_dir)
        list_img_cv = []
        type_list = []
        imgs = []
        for img_file in list_imgs:
            abs_dir = os.path.join(data_dir, img_file)
            img_cv = cv2.imread(abs_dir)
            img_cv = cv2.resize(img_cv, (160, 160))

            face_pixels = img_cv.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std

            img_np = np.asarray([face_pixels])
            imgs.append(img_np)
        imgs = np.array(imgs)
        imgs = np.squeeze(imgs)
        embedding_vectors = embedding_model.predict(imgs)
        predicts = KNN_model.find_item(embedding_vectors)
        for predict in predicts:
            if i - int(predict) == 0:
                correct += 1
            count += 1
    return correct / count


def evaluate_sample(file_path, embedding_model, KNN_model):
    label_dict = load_label_dict()
    img_cv = cv2.imread(file_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    img_np = np.asarray([img_cv])
    embedding_vector = embedding_model.predict(img_np)
    predict = KNN_model.find_item(embedding_vector)
    return predict
