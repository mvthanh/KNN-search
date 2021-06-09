import Augmentor
import os
import sys
import argparse


def augment(input_path, output_path):
    list_dir = os.listdir(input_path)
    # print(list_dir)
    for dir_ in list_dir:
        print(os.path.join(input_path, dir_))
        p = Augmentor.Pipeline(os.path.join(input_path, dir_))
        # p.ground_truth(os.path.join(path,dir_))
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        # p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        # p.flip_top_bottom(probability=0.5)
        p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
        p.sample(1000)
        sub_dirs = os.listdir(os.path.join(input_path, dir_, "output"))
        os.makedirs(os.path.join(output_path, dir_))
        for sub_dir in sub_dirs:
            os.rename(os.path.join(input_path, dir_, "output", sub_dir), os.path.join(output_path, dir_, sub_dir))
        os.rmdir(os.path.join(input_path, dir_, "output"))


def main(argv):
    parser = argparse.ArgumentParser(description='Processing data !.')
    parser.add_argument('-ip', '--input', help='Input data path', type=str)
    parser.add_argument('-op', '--output', help='Output data path ', type=str)

    args = parser.parse_args()
    augment(args.input, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
