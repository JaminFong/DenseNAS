import torch
import numpy as np
import time

class data_prefetcher():
    def __init__(self, loader, mean=None, std=None, is_cutout=False, cutout_length=16):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if mean is None:
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        else:
            self.mean = torch.tensor([m * 255 for m in mean]).cuda().view(1,3,1,1)
        if std is None:
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        else:
            self.std = torch.tensor([s * 255 for s in std]).cuda().view(1,3,1,1)
        self.is_cutout = is_cutout
        self.cutout_length = cutout_length
        self.preload()

    def normalize(self, data):
        data = data.float()
        data = data.sub_(self.mean).div_(self.std)
        return data
    
    def cutout(self, data):
        batch_size, h, w = data.shape[0], data.shape[2], data.shape[3]
        mask = torch.ones(batch_size, h, w).cuda()
        y = torch.randint(low=0, high=h, size=(batch_size,))
        x = torch.randint(low=0, high=w, size=(batch_size,))
        
        y1 = torch.clamp(y - self.cutout_length // 2, 0, h)
        y2 = torch.clamp(y + self.cutout_length // 2, 0, h)
        x1 = torch.clamp(x - self.cutout_length // 2, 0, w)
        x2 = torch.clamp(x + self.cutout_length // 2, 0, w)
        for i in range(batch_size):
            mask[i][y1[i]: y2[i], x1[i]: x2[i]] = 0.
        mask = mask.expand_as(data.transpose(0,1)).transpose(0,1)
        data *= mask
        return data

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.normalize(self.next_input)
            if self.is_cutout:
                self.next_input = self.cutout(self.next_input)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets
