import hnswlib
import os


class Jugde():
    def __init__(self, path):
        self.dim_KNN = 128
        self.KNN_DIR = path
        self.KNN_searcher = self.build_KNN(path)
        # self.product_dict = {}

    def build_KNN(self, KNN_DIR="./KNN_data/knn_searcher.bin"):
        self.KNN_DIR = KNN_DIR
        if not os.path.exists(KNN_DIR):
            p = hnswlib.Index(space='cosine', dim=self.dim_KNN)
            p.init_index(max_elements=10000, ef_construction=150, M=16)
            p.set_ef(10)
            p.set_num_threads(4)
            return p
        else:
            p = hnswlib.Index(space='cosine', dim=self.dim_KNN)
            p.load_index(KNN_DIR)
            return p

    def add_items(self, feature):
        vectors, labels = feature
        self.KNN_searcher.add_items(vectors, labels)
        print(len(vectors), "iteme(s) added")

    def find_item(self, vector, no_nearest=1):
        labels, distance = self.KNN_searcher.knn_query(vector, k=no_nearest)
        return labels

    def save_KNN(self):
        self.KNN_searcher.save_index(self.KNN_DIR)
