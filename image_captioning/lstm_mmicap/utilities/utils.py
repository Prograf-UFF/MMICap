import torch
import os, re, json

import torch.distributed as dist
import utils

from pycocotools.coco import COCO
from .pycocoevalcap.eval import COCOEvalCap


## Annotation Ground Truth
ANNOTATION_VAL_GT = "annotation_val_gt.json"
ANNOTATION_TEST_GT = "annotation_test_gt.json"


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0.
        self.max_cider_val = 0.# float('inf')

    def early_stop(self, cider_val):
        if cider_val > self.max_cider_val:
            self.max_cider_val = cider_val
            self.counter = 0
        elif cider_val < (self.max_cider_val + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption
    
def load_checkpoint(model,filename):
    if os.path.isfile(filename):        
        checkpoint = torch.load(filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%filename)  
    return model,msg


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    try: dist.barrier()
    except: pass #print("NOT used distribuited..")

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file


def coco_caption_eval(coco_gt_root:str, results_file:str, split:str, use_spice:bool=False):   
    """ """
    filenames = {'val':ANNOTATION_VAL_GT,'test':ANNOTATION_TEST_GT}
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate(use_spice)

    # print output evaluation scores
    #for metric, score in coco_eval.eval.items():
    #    print(f'{metric}: {score:.3f}')
    
    return coco_eval