import argparse
import os
import os.path as osp
import sys

import cv2
import lmdb
import msgpack
import numpy as np
from PIL import Image
from tqdm import tqdm

image_size=None

class Datum(object):
    def __init__(self, shape=None, image=None, label=None):
        self.shape = shape
        self.image = image
        self.label = label

    def SerializeToString(self, img=None):
        image_data = self.image.astype(np.uint8).tobytes()
        label_data = np.uint16(self.label).tobytes()
        return msgpack.packb(image_data+label_data, use_bin_type=True)

    def ParseFromString(self, raw_data, orig_img):
        raw_data = msgpack.unpackb(raw_data, raw=False)
        raw_img_data = raw_data[:-2]
        # share the memory of data while fromstring copy one
        image_data = np.frombuffer(raw_img_data, dtype=np.uint8)
        self.image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        raw_label_data = raw_data[-2:]
        self.label = np.frombuffer(raw_label_data, dtype=np.uint16)


def create_dataset(output_path, image_folder, image_list, image_size):
    image_name_list = [i.strip() for i in open(image_list)]
    n_samples = len(image_name_list)
    env = lmdb.open(output_path, map_size=1099511627776, meminit=False, map_async=True) # 1TB

    txn = env.begin(write=True)
    classes = [d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]
    for idx, image_name in enumerate(tqdm(image_name_list)):
        image_path = os.path.join(image_folder, image_name)
        label_name = image_name.split('/')[0]
        label = classes.index(label_name)
        if not os.path.isfile(image_path):
            raise RuntimeError('%s does not exist' % image_path)

        img = cv2.imread(image_path)
        img_orig = img

        if image_size:
            resize_ratio = float(image_size)/min(img.shape[0:2])
            new_size = (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)) #inverse order for cv2
            img = cv2.resize(src=img, dsize=new_size)
        img = cv2.imencode('.JPEG', img)[1]

        image = np.asarray(img)
        datum = Datum(image.shape, image, label)
        txn.put(image_name.encode('ascii'), datum.SerializeToString())

        if (idx + 1) % 1000 == 0:
            txn.commit() 
            txn = env.begin(write=True)
    txn.commit()
    env.sync()
    env.close()

    print(f'Created dataset with {n_samples:d} samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--image_size', type=int, default=None, help='the size of the image u want to pack')
    parser.add_argument('--image_path', type=str, default='', help='the path of the images')
    parser.add_argument('--list_path', type=str, default='', help='the path of the image list')
    parser.add_argument('--output_path', type=str, default='', help='the output path of the lmdb file')
    parser.add_argument('--split', type=str, default='', help='the split path of the images: train / val')
    args = parser.parse_args()

    image_size = args.image_size if args.image_size else image_size
    image_folder = osp.join(args.image_path, args.split)
    image_list = osp.join(args.list_path, '{}_datalist'.format(args.split))

    if image_size:
        output_path = osp.join(args.output_path, '{}_{}'.format(args.split, image_size))
    else:
        output_path = osp.join(args.output_path, args.split)

    create_dataset(output_path, image_folder, image_list, image_size)
