from keras.models import load_model


def load_facenet():
    model = load_model('../resources/facenet_keras.h5')
    return model


if __name__ == '__main__':
    model = load_facenet()
    model.summary()
