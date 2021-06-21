from keras.models import load_model


def load_facenet():
    model = load_model(r'C:\Users\TechCare\Desktop\KNN\KNN-search\resources\facenet_keras.h5')
    return model


if __name__ == '__main__':
    model = load_facenet()
    model.summary()
