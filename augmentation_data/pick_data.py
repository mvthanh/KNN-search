import os
from tqdm import tqdm
import shutil

NUM_IMG = 100


def copy_file(src, dst):
    shutil.copy(src, dst)


def list_file(dir_path):
    files = os.listdir(dir_path)
    files = list(sorted(files, key=lambda x: int(x.split('.')[0])))
    return files


def main():
    in_dir = r'D:\BAP\data_BAP9'
    out_dir = r'D:\BAP\Data_out_t9'

    for dir in tqdm(os.listdir(in_dir)):
        img_dir = os.path.join(in_dir, dir)
        out_img_dir = os.path.join(out_dir, dir)
        try:
            os.mkdir(out_img_dir)
        except:
            pass
        n = len(os.listdir(img_dir)) // NUM_IMG

        for ids, file in enumerate(list_file(img_dir)):
            if ids % n != 0:
                continue
            file_path = os.path.join(img_dir, file)
            copy_file(file_path, out_img_dir)


if __name__ == '__main__':
    main()
