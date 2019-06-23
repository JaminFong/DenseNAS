import os

import torch
import torchvision
import torchvision.transforms as vision_transforms

from dataset import lmdb_dataset
from dataset import torchvision_extension as vision_transforms_extension

meanstd = {
   'mean':[0.485, 0.456, 0.406],
   'std': [0.229, 0.224, 0.225],
}

pca = {
   'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
   'eigvec': torch.Tensor([
      [-0.5675,  0.7192,  0.4009],
      [-0.5808, -0.0045, -0.8140],
      [-0.5836, -0.6948,  0.4203],
   ])
}


class ImageNet12(object):

    def __init__(self, trainFolder, testFolder, num_workers=8, pin_memory=True, 
                size_images=224, scaled_size=256, data_config=None):
        self.data_config = data_config
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_dataset = self.data_config.patch_dataset
        self.meanstd = meanstd
        self.pca = pca

        #images will be rescaled to match this size
        if not isinstance(size_images, int):
            raise ValueError('size_images must be an int. It will be scaled to a square image')
        self.size_images = size_images
        self.scaled_size = scaled_size


    def _getTransformList(self, aug_type):

        assert aug_type in ['rand_scale', 'random_sized', 'week_train', 'validation']
        list_of_transforms = []

        if aug_type == 'validation':
            list_of_transforms.append(vision_transforms.Resize(self.scaled_size))
            list_of_transforms.append(vision_transforms.CenterCrop(self.size_images))
            list_of_transforms.append(vision_transforms.ToTensor())
            list_of_transforms.append(vision_transforms.Normalize(mean=self.meanstd['mean'],
                                                                std=self.meanstd['std']))
        
        return vision_transforms.Compose(list_of_transforms)


    def _getTestSet(self):
        # first we define the training transform we will apply to the dataset

        test_transform = self._getTransformList('validation')

        if self.data_config.val_data_type == 'img':
            test_set = torchvision.datasets.ImageFolder(self.testFolder, test_transform)
        elif self.data_config.val_data_type == 'lmdb':
            test_set = lmdb_dataset.ImageFolder(self.testFolder, 
                            os.path.join(self.testFolder, '..', 'val_datalist'),
                            test_transform)
            self.test_num_examples = test_set.__len__()

        return test_set


    def getTestLoader(self, batch_size, shuffle=False):
        
        test_set = self._getTestSet()
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                batch_size=batch_size,
                                                shuffle=shuffle, 
                                                num_workers=self.num_workers, 
                                                pin_memory=self.pin_memory)
        return test_loader
