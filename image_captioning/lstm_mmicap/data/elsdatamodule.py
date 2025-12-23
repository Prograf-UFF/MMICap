#imports 
import os, json, re
from collections import Counter
import spacy
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T

from PIL import Image
from utilities.utils import pre_caption

#using spacy for the better text tokenization 
spacy_eng = spacy.load('en_core_web_sm')

## Annotation trainig
ANNOTATION_TRAIN = "elscap_train.json"
ANNOTATION_TEST = "elscap_test.json"
ANNOTATION_VAL = "elscap_val.json"
TRAIN_NAME = "train"
VAL_NAME = "val"
TEST_NAME = "test"
MAX_WORDS_PARAGRAPH = 512


class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in tqdm(sentence_list, desc="      Build vocabulary"):
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]


class ElsCapDataset_Train(Dataset):
    """
    ElsCapDataset Train
    """
    def __init__(self,image_root:str,ann_root:str,transform=None,freq_threshold=5):
        self.image_root = image_root
        self.annotation = json.load(open(os.path.join(ann_root,ANNOTATION_TRAIN),'r'))
        self.transform = transform

        self.img_ids = {}  
        captions_voc = [] # TODO
        for idx, ann in enumerate(self.annotation):
            img_id = ann['path'][:-4]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = idx 
            captions_voc.append(ann["caption"])
            #captions_voc.append(ann["paragraph"])
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(captions_voc)

        del captions_voc
        
    
    def __len__(self):
        return len(self.annotation)
    
    def get_vocab(self):
        return self.vocab
    
    def __getitem__(self,idx):
        ann = self.annotation[idx]
        
        image_path = os.path.join(self.image_root, TRAIN_NAME, ann['path'])     
        img = Image.open(image_path).convert('RGB')   
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(pre_caption(ann['caption'])) # TODO: maximo 512
        caption_vec += [self.vocab.stoi["<EOS>"]]

        paragraph = "sumarize: "+pre_caption(ann['paragraph'], MAX_WORDS_PARAGRAPH)
        
        return img, torch.tensor(caption_vec), paragraph #, self.img_ids[ann['path'][:-4]]

class ElsCapDataset_Eval(Dataset):
    """
    ElsCapDataset Eval
    """
    def __init__(self,image_root:str,ann_root:str,split:str, vocab, transform=None, freq_threshold=5):
        self.image_root = image_root
        self.transform = transform

        filenames = {'val':ANNOTATION_VAL,'test':ANNOTATION_TEST}
        self.val_test = VAL_NAME if split=='val' else TEST_NAME
        
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r')) 
        self.vocab = vocab
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self,idx):
        ann = self.annotation[idx]
        
        image_path = os.path.join(self.image_root, self.val_test, ann['path'])        
        img = Image.open(image_path).convert('RGB')   
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        img_id = ann['path'][:-4]
        paragraph = "sumarize: "+pre_caption(ann['paragraph'], MAX_WORDS_PARAGRAPH) 
        return img, img_id, paragraph


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        paragraphs = [item[2] for item in batch]
        return imgs,targets,paragraphs


# BASE: https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k
if __name__=="__main__":
    #defing the transform to be applied
    transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])

    #testing the dataset class
    dataset =  ElsCapDataset_Train(
        image_root = "../dataset/ElsCap/imgs",
        ann_root = "./annotation/blip",
        transform=transforms
    )

    #token to represent the padding
    pad_idx = dataset.get_vocab().stoi["<PAD>"]

    img, caps, parag = dataset[0]
    #show_image(img,"Image")
    print("Token:",caps, "PARAGRAPH: ", parag)
    print("Sentence:")
    print([dataset.vocab.itos[token] for token in caps.tolist()])

    BATCH_SIZE = 4
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )

    #generating the iterator from the dataloader
    dataiter = iter(data_loader)

    #getting the next batch
    batch = next(dataiter)

    #unpacking the batch
    images, captions, paras = batch

    #showing info of image in single batch
    for i in range(BATCH_SIZE):
        _,cap, para = images[i],captions[i], paras[i]
        caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
        print(len(caption_label), len(para))