import os

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
from PIL import Image

import lmdb


class Datum(object):
    def __init__(self, shape=None, image=None, label=None):
        self.shape = shape
        self.image = image
        self.label = label

    def SerializeToString(self):
        image_data = self.image.astype(np.uint8).tobytes()
        label_data = np.uint16(self.label).tobytes()
        return msgpack.packb(image_data+label_data, use_bin_type=True)

    def ParseFromString(self, raw_data):
        raw_data = msgpack.unpackb(raw_data, raw=False)
        raw_img_data = raw_data[:-2]
        image_data = np.frombuffer(raw_img_data, dtype=np.uint8) #share the memory of data while from string copy one
        self.image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)        

        raw_label_data = raw_data[-2:]
        self.label = np.frombuffer(raw_label_data, dtype=np.uint16)


class DatasetFolder(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, list_path, transform=None, target_transform=None, patch_dataset=False):
        self.root = root
        self.patch_dataset = patch_dataset
        
        if patch_dataset:
            self.txn = []
            for path in os.listdir(root):
                lmdb_path = os.path.join(root, path)
                if os.path.isdir(lmdb_path):
                    env = lmdb.open(lmdb_path, 
                                readonly=True, 
                                lock=False, 
                                readahead=False, 
                                meminit=False)
                    txn = env.begin(write=False)
                    self.txn.append(txn)

        else:
            self.env = lmdb.open(root,
                                readonly=True,
                                lock=False, 
                                readahead=False,
                                meminit=False)
            self.txn = self.env.begin(write=False)

        self.list_path = list_path
        self.samples = [image_name.strip() for image_name in open(list_path)]

        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        img_name = self.samples[index]

        if self.patch_dataset:
            txn_index = index // (len(self.samples) // 10)
            if txn_index==10:
                txn_index = 9
            txn = self.txn[txn_index]
        else:
            txn = self.txn

        datum = Datum()
        data_bin = txn.get(img_name.encode('ascii'))
        if data_bin is None:
            raise RuntimeError(f'Key {img_name} not found')
        datum.ParseFromString(data_bin)

        sample = Image.fromarray(cv2.cvtColor(datum.image, cv2.COLOR_BGR2RGB))
        target = np.int(datum.label)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolder(DatasetFolder):
    def __init__(self, root, list_path, transform=None, target_transform=None, patch_dataset=False):
        super(ImageFolder, self).__init__(root, list_path,
                                          transform=transform,
                                          target_transform=target_transform,
                                          patch_dataset=patch_dataset)
        self.imgs = self.samples
