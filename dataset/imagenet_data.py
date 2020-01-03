import os

import torch
import torchvision
import torchvision.transforms as transforms

from . import lmdb_dataset
from . import torchvision_extension as transforms_extension
from .prefetch_data import fast_collate

class ImageNet12(object):

    def __init__(self, trainFolder, testFolder, num_workers=8, pin_memory=True, 
                size_images=224, scaled_size=256, type_of_data_augmentation='rand_scale', 
                data_config=None):

        self.data_config = data_config
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_dataset = self.data_config.patch_dataset

        #images will be rescaled to match this size
        if not isinstance(size_images, int):
            raise ValueError('size_images must be an int. It will be scaled to a square image')
        self.size_images = size_images
        self.scaled_size = scaled_size

        type_of_data_augmentation = type_of_data_augmentation.lower()
        if type_of_data_augmentation not in ('rand_scale', 'random_sized'):
            raise ValueError('type_of_data_augmentation must be either rand-scale or random-sized')
        self.type_of_data_augmentation = type_of_data_augmentation


    def _getTransformList(self, aug_type):

        assert aug_type in ['rand_scale', 'random_sized', 'week_train', 'validation']
        list_of_transforms = []

        if aug_type == 'validation':
            list_of_transforms.append(transforms.Resize(self.scaled_size))
            list_of_transforms.append(transforms.CenterCrop(self.size_images))

        elif aug_type == 'week_train':
            list_of_transforms.append(transforms.Resize(256))
            list_of_transforms.append(transforms.RandomCrop(self.size_images))
            list_of_transforms.append(transforms.RandomHorizontalFlip())

        else:
            if aug_type == 'rand_scale':
                list_of_transforms.append(transforms_extension.RandomScale(256, 480))
                list_of_transforms.append(transforms.RandomCrop(self.size_images))
                list_of_transforms.append(transforms.RandomHorizontalFlip())

            elif aug_type == 'random_sized':
                list_of_transforms.append(transforms.RandomResizedCrop(self.size_images, 
                                        scale=(self.data_config.random_sized.min_scale, 1.0)))
                list_of_transforms.append(transforms.RandomHorizontalFlip())

            if self.data_config.color:
                list_of_transforms.append(transforms.ColorJitter(brightness=0.4,
                                                                contrast=0.4,
                                                                saturation=0.4))
        return transforms.Compose(list_of_transforms)


    def _getTrainSet(self):

        train_transform = self._getTransformList(self.type_of_data_augmentation)

        if self.data_config.train_data_type == 'img':
            train_set = torchvision.datasets.ImageFolder(self.trainFolder, train_transform)
        elif self.data_config.train_data_type == 'lmdb':
            train_set = lmdb_dataset.ImageFolder(self.trainFolder, 
                                os.path.join(self.trainFolder, '..', 'train_datalist'),
                                train_transform,
                                patch_dataset=self.patch_dataset)
        self.train_num_examples = train_set.__len__()
            
        return train_set


    def _getWeekTrainSet(self):

        train_transform = self._getTransformList('week_train')
        if self.data_config.train_data_type == 'img':
            train_set = torchvision.datasets.ImageFolder(self.trainFolder, train_transform)
        elif self.data_config.train_data_type == 'lmdb':
            train_set = lmdb_dataset.ImageFolder(self.trainFolder, 
                                os.path.join(self.trainFolder, '..', 'train_datalist'),
                                train_transform,
                                patch_dataset=self.patch_dataset)
        self.train_num_examples = train_set.__len__()
        return train_set


    def _getTestSet(self):

        test_transform = self._getTransformList('validation')
        if self.data_config.val_data_type == 'img':
            test_set = torchvision.datasets.ImageFolder(self.testFolder, test_transform)
        elif self.data_config.val_data_type == 'lmdb':
            test_set = lmdb_dataset.ImageFolder(self.testFolder, 
                            os.path.join(self.testFolder, '..', 'val_datalist'),
                            test_transform)
            self.test_num_examples = test_set.__len__()
        return test_set


    def getTrainLoader(self, batch_size, shuffle=True):
        
        train_set = self._getTrainSet()
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory, 
            sampler=None, collate_fn=fast_collate)
        return train_loader


    def getWeekTrainLoader(self, batch_size, shuffle=True):
        
        train_set = self._getWeekTrainSet()
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size,
                                                shuffle=shuffle, 
                                                num_workers=self.num_workers, 
                                                pin_memory=self.pin_memory,
                                                collate_fn=fast_collate)
        return train_loader


    def getTestLoader(self, batch_size, shuffle=False):
        
        test_set = self._getTestSet()

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory, sampler=None,
            collate_fn=fast_collate)
        return test_loader


    def getTrainTestLoader(self, batch_size, train_shuffle=True, val_shuffle=False):
        
        train_loader = self.getTrainLoader(batch_size, train_shuffle)
        test_loader = self.getTestLoader(batch_size, val_shuffle)
        return train_loader, test_loader


    def getSetTrainTestLoader(self, batch_size, train_shuffle=True, val_shuffle=False):

        train_loader = self.getTrainLoader(batch_size, train_shuffle)
        week_train_loader = self.getWeekTrainLoader(batch_size, train_shuffle)
        test_loader = self.getTestLoader(batch_size, val_shuffle)
        return (train_loader, week_train_loader), test_loader
