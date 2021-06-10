import tensorflow as tf
from build_knn.train_KNN import *
from model_ebd.load_model import load_facenet
import sys
import argparse
from build_knn.KNN_model import Jugde


def evaluate(embedding_model, KNN_model, test_dir):
    print("Evaluating folder", test_dir + "...")
    acc = evaluate_folder(test_dir, embedding_model, KNN_model)
    print("The accuracy of this whole model is", acc)


def add_data(embedding_model, KNN_model, add_dir):
    train_KNN_folder(add_dir, embedding_model, KNN_model)


def evaluate_single(embedding_model, KNN_model, test_dir):
    label = evaluate_sample(test_dir, embedding_model, KNN_model)
    if (label != -1):
        print("This is a", label)
    else:
        print("Something wrong dude")


def main(argv):
    parser = argparse.ArgumentParser(description='Processing data !.')
    parser.add_argument('-ebdp', '--embedding_path', help='Embedding model directory', type=str)
    parser.add_argument('-knnp', '--KNN_path', help='KNN directory ', type=str)
    parser.add_argument('-td', '--test_dir', help='Directory of data that you want to test ', type=str, default="Null")
    parser.add_argument('-ad', '--add_data_dir', help='Directory of data that you want to add ', type=str,
                        default="Null")
    parser.add_argument('-es', '--test_single', help='Directory of picture you want to predict label ', type=str,
                        default="Null")

    args = parser.parse_args()
    KNN_model = Jugde(args.KNN_path)

    latest = tf.train.latest_checkpoint(args.embedding_path)
    model = load_facenet()
    model.load_weights(latest).expect_partial()

    if args.test_dir != "Null":
        print("wtf")
        evaluate(model, KNN_model, args.test_dir)
    if args.add_data_dir != "Null":
        add_data(model, KNN_model, args.add_data_dir)
    if args.test_single != "Null":
        evaluate_single(model, KNN_model, args.test_single)


if __name__ == "__main__":
    main(sys.argv[1:])
