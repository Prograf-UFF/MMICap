import os
import json
from PIL import Image
from torch.utils.data import Dataset
from utilities.utils import pre_caption

ANNOTATION_TRAIN='elscap_train.json'
ANNOTATION_VAL='elscap_val.json'
ANNOTATION_TEST='elscap_test.json'
MAX_WORDS_PARAGRAPH = 512


class elsdataset_train(Dataset):
    def __init__(self, transform, image_root:str, ann_root:str, max_words:int=30, prompt:str=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        self.annotation = json.load(open(os.path.join(ann_root,ANNOTATION_TRAIN),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        for idx, ann in enumerate(self.annotation):
            img_id = ann['path'][:-4]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = idx    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index:int):    
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, 'train', ann['path'])     
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        paragraph = "sumarize: "+pre_caption(ann['paragraph'], MAX_WORDS_PARAGRAPH) 

        return image, caption, paragraph, self.img_ids[ann['path'][:-4]] 
    
    
class elsdataset_caption_eval(Dataset):
    def __init__(self, transform, image_root:str, ann_root:str, split:str):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':ANNOTATION_VAL,'test':ANNOTATION_TEST}
        self.val_test = 'val' if split=='val' else 'test'
        
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))   
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index:int):    
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, self.val_test, ann['path'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        #img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        img_id = ann['path'][:-4]
        paragraph = "sumarize: "+pre_caption(ann['paragraph'], MAX_WORDS_PARAGRAPH) 
        
        return image, img_id , paragraph #int(img_id)