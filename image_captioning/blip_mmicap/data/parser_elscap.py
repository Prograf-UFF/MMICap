import json
import os
from os.path import join, basename
from tqdm import tqdm
from glob import glob
from typing import List
import argparse

ROOT_PROJECT = "."
ROOT_DATASET = "."
ANNOTATIONS_PATH = os.path.join(ROOT_PROJECT, "annotation", "blip")
DATASET_PATH = os.path.join(ROOT_DATASET, "dataset", "ElsCap")


def create_annotations(data_path: str, out_file:str, is_train:bool=True):
    filename_list = glob(os.path.join(DATASET_PATH, 'imgs', data_path, "*.jpg"))
    res = []

    for filename in tqdm(filename_list):
        with open(os.path.join(DATASET_PATH, 'info', data_path, basename(filename)[:-4]+".json"), 'r') as f:
            data = json.load(f)
        caption = data["figure"]["caption"].encode('ascii','ignore').decode('utf-8').strip().replace('\n', '')
        paragraphs = ''
        # get paragraphs content
        for val in data['paragraphs']:
            paragraphs += val['text'].encode('ascii','ignore').decode('utf-8').strip().replace('\n', '')
        #count = count_words(caption)
        #count_paragraphs = count_words(paragraphs)
        
        #if count>=MIN_WORDS and count_paragraphs>=MIN_WORDS:
        if is_train:
            res.append({
                "path": basename(filename), 
                "paragraph": paragraphs,
                "caption": caption
            })
        else:
            res.append({
                "path": basename(filename), 
                "paragraph": paragraphs,
            })
            
    with open(out_file, 'w', encoding='utf8') as json_file:
        json.dump(res, json_file, ensure_ascii=False)


def create_annotations_gt(data_path: str, out_file:str):
    filename_list = glob(os.path.join(DATASET_PATH, 'imgs', data_path, "*.jpg"))
    annotations = []
    images = []

    for filename in tqdm(filename_list):
        with open(os.path.join(DATASET_PATH, 'info', data_path, basename(filename)[:-4]+".json"), 'r') as f:
            data = json.load(f)
        caption = data["figure"]["caption"].encode('ascii','ignore').decode('utf-8').strip().replace('\n', '')
        paragraphs = ''
        # get paragraphs content
        for val in data['paragraphs']:
            paragraphs += val['text'].encode('ascii','ignore').decode('utf-8').strip().replace('\n', '')
        #count = count_words(caption)
        #count_paragraphs = count_words(paragraphs)
        
        #if count>=MIN_WORDS and count_paragraphs>=MIN_WORDS:
        id_img = basename(filename)[:-4]
        annotations.append({"image_id": id_img, "caption": caption, "id": id_img})
        images.append({"id": id_img})
            
    with open(out_file, 'w', encoding='utf8') as file:
        data = {
            'annotations': annotations,
            'images': images,
            }
        json.dump(data, file, ensure_ascii=False)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_path', default=None)
    #parser.add_argument('--out_path', default=ANNOTATIONS_PATH)
    #args = parser.parse_args()

    paths = ['test','train','val'] #OPTIONS ['test','train','val']

    if not os.path.exists(ANNOTATIONS_PATH): os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
    for p in paths:
        out_json = os.path.join(ANNOTATIONS_PATH, f"elscap_{p}.json")
        print(f"CREATE PARSE DATASET to {p} and save in {out_json} ..")
        is_train = True if p=="train" else False
        create_annotations(p, out_json, is_train)
    
    # Create GT test
    out_json = join(ROOT_PROJECT,"annotation","elscap_gt","annotation_test_gt.json")
    if not os.path.exists(os.path.dirname(out_json)): os.makedirs(os.path.dirname(out_json), exist_ok=True)
    create_annotations_gt('test', out_json)

    print("Finish ..")