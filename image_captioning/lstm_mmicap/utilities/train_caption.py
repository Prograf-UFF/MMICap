#imports
import numpy as np
import torch, wandb
import argparse
import os, random, json
import yaml, time, datetime
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from data import create_dataset, create_sampler, create_loader
from data.elsdatamodule import CapsCollate
from utilities.utils import EarlyStopper, save_result, coco_caption_eval, load_checkpoint
import utils
from utils import cosine_lr_schedule
from nn.model import EncoderDecoder

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

USE_WANDB = True
WANDB_PROJECT_NAME = "lstm_att_our"   # "mmicap_scicapplus"
DATASET_NAME = 'elsdataset' # 'scicapplus' | 'elsdataset'


def train(model, data_loader, optimizer, epoch, device, criterion, vocab_size, print_freq = 100):
    # train
    model.train()  

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    for idx, (image, captions, paragraphs) in enumerate(metric_logger.log_every(iter(data_loader), print_freq, 'Train Caption Epoch: [{}]'.format(epoch))):
        image,captions = image.to(device),captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs,attentions = model(image, captions, paragraphs)

        # Calculate the batch loss.
        targets = captions[:,1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        
        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        del image
        del captions
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    res = {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    if utils.is_main_process() and USE_WANDB: wandb.log({"loss": float(res['loss'])})         # WANDB  <---    
    return res   


@torch.no_grad()
def evaluate(model, data_loader, device, config, vocab, print_freq = 100):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")

    result = []
    for img, image_id, paragraphs in metric_logger.log_every(data_loader, print_freq, '   Caption generation:'): 
        
        batch = img.shape[0]
        captions = []
        features = model.encoder(img.to(device))
        #print("----> ",img.shape, img[1].shape, " -- ", features.shape, features[1].unsqueeze(0).shape)
        for idx in range(batch):
            caps,alphas = model.decoder.generate_caption(features[idx].unsqueeze(0), paragraphs[idx],vocab=vocab, max_len=config['max_length'])
            caption = ' '.join(caps)
            result.append({"image_id": image_id[idx], "caption": caption})

        #for caption, img_id in zip(captions, image_id):
        #    result.append({"image_id": img_id, "caption": caption})
  
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

    vocab = train_dataset.get_vocab()
    #token to represent the padding
    pad_idx = vocab.stoi["<PAD>"]

    NUM_WORKER = 4
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[NUM_WORKER,NUM_WORKER,1],
                                                          is_trains=[True, False, False], collate_fns=[CapsCollate(pad_idx=pad_idx,batch_first=True),None,None])
    #vocab_size
    vocab_size = len(vocab)

    #init model
    model = EncoderDecoder(
        embed_size=config['embed_size'],
        vocab_size = vocab_size,
        attention_dim=config['attention_dim'],
        encoder_dim=config['encoder_dim'],
        decoder_dim=config['decoder_dim']
    )
    if config['pretrained']:
        model,msg = load_checkpoint(model,config['pretrained'])
        assert(len(msg.missing_keys)==0)

    model = model.to(device)   
    model_without_ddp = model   
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
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
            train_stats = train(model, train_loader, optimizer, epoch, device, criterion, vocab_size)
        
        val_result = evaluate(model_without_ddp, val_loader, device, config, vocab)  
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id') 

        if utils.is_main_process():   
            coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
            
            if args.evaluate:  
                test_result = evaluate(model_without_ddp, test_loader, device, config, vocab)  
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_elscap_base.yaml')
    parser.add_argument('--output_dir', default='./output/lstm_att_our/elscap')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
    pass
