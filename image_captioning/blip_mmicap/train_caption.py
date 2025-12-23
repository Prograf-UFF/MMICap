'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from utilities.utils import save_result, coco_caption_eval

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

USE_WANDB = True
WANDB_PROJECT_NAME = "mmicap_elscap"   # "mmicap_scicapplus"
DATASET_NAME = 'elsdataset' # 'scicapplus' | 'elsdataset'

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


def train(model, data_loader, optimizer, epoch, device, print_freq = 100):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    loss_sum = 0
    for n, (image, caption, paragraph, _) in enumerate(metric_logger.log_every(data_loader, print_freq, 'Train Caption Epoch: [{}]'.format(epoch))):
        image = image.to(device)  
        loss = model(image, caption, paragraph)      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        loss_sum = loss_sum + loss.item()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg()) 
    res = {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    if utils.is_main_process() and USE_WANDB: wandb.log({"loss": loss_sum/(n+1)})         # WANDB  <---    
    return res   


@torch.no_grad()
def evaluate(model, data_loader, device, config, print_freq = 100):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")

    result = []
    for image, image_id, paragraph in metric_logger.log_every(data_loader, print_freq, '   Caption generation:'): 
        image = image.to(device)       
        captions = model.generate(image, paragraph, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            #result.append({"image_id": img_id.item(), "caption": caption})
            result.append({"image_id": img_id, "caption": caption})
  
    return result


def main(args, config):
    utils.init_distributed_mode(args)   
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(DATASET_NAME, config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,1],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'], med_config=config['med_config'])

    model = model.to(device)   
    model_without_ddp = model   
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module 

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=float(config['init_lr']), weight_decay=float(config['weight_decay']))
    
    print("Start training")
    start_time = time.time() 
    best, best_epoch, cider_val = 0, 0, 0
    early_stopper = EarlyStopper(patience=4, min_delta=0)

    if utils.is_main_process() and USE_WANDB: wandb.init(project=WANDB_PROJECT_NAME)

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:    
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, int(config['max_epoch']), float(config['init_lr']), float(config['min_lr']))
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        
        val_result = evaluate(model_without_ddp, val_loader, device, config)  
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')        
  
        if utils.is_main_process():   
            coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
            
            if args.evaluate:  
                test_result = evaluate(model_without_ddp, test_loader, device, config)  
                test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  
                coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test') 

                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                if USE_WANDB: wandb.log({**{f'val_{k}': v for k, v in coco_val.eval.items()}})       # WANDB  <---
                cider_val = coco_val.eval['CIDEr']
                if cider_val > best:
                    best = cider_val
                    best_epoch = epoch                
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_best.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},                       
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: break
        try: dist.barrier()
        except: pass #print("NOT used distribuited..")
        # early stopping
        if early_stopper.early_stop(cider_val): break   

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


def start_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_our_elscap_base.yaml')
    parser.add_argument('--output_dir', default='/home/josehuillca/Documents/Github/Doutorado_textSummarization/output/blip_our/elscap')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)

    # Vit-Large: pretrained/model_large_caption.pth
    # Vit-Base: pretrained/model_base_caption_capfilt_large.pth


if __name__ == '__main__':
    start_training()
    print("finish..")