import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transform.randaugment import RandomAugment
from .elscap_dataset import elsdataset_train, elsdataset_caption_eval
from .scicapplus_dataset import scicapplus_train, scicapplus_eval


import numbers, numpy as np
from torchvision.transforms.functional import pad
class NewPad(object):
    def __init__(self, fill:int=255, padding_mode:str='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode
    
    def get_padding(self,image):    
        w, h = image.size
        max_wh = np.max([w, h])
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        return padding
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        return pad(img, self.get_padding(img), self.fill, self.padding_mode)
    

def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])   
    # transform_train = transforms.Compose([
    #         NewPad(),                        
    #         transforms.Resize((config['image_size'],config['image_size']), interpolation=InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(),    
    #         transforms.ToTensor(),
    #         normalize,
    #     ])      
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='elsdataset':   
        train_dataset = elsdataset_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = elsdataset_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = elsdataset_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset
    if dataset=='scicapplus':   
        train_dataset = scicapplus_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = scicapplus_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = scicapplus_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
