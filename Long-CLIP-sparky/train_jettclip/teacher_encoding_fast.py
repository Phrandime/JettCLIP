import json
from PIL import Image
import torch
import torch.utils.data as data
import os
from tqdm import tqdm
import pickle
from multiprocessing import cpu_count

import sys
sys.path.append("..")

from model import longclip

data4v_root = '../../Long-CLIP/ShareGPT4V/data/'  # 'sharegpt4v/data/'
json_name = 'share-captioner_coco_lcs_676k_1107.json'
image_root = '../../Long-CLIP/ShareGPT4V/data/'  # 'sharegpt4v/data/'

model_ckpt = '../checkpoints/longclip-B.pt'
assert torch.cuda.is_available()

# Custom Dataset
class ImageTextDataset(data.Dataset):
    def __init__(self, json_path, image_root):
        with open(json_path, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:400]
        self.image_root = image_root

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        data = self.json_data[idx]
        caption = data['conversations'][1]['value'].replace("\n", " ")
        caption_short = caption.split(". ")[0]
        image_path = os.path.join(self.image_root, data['image'])
        image = Image.open(image_path)
        return {
            'id': idx,
            'image': image,
            'caption': caption,
            'caption_short': caption_short,
            'image_path': data['image']
        }

# Batch processing
def process_batch(model, preprocess, batch):
    images = torch.stack([preprocess(item['image']) for item in batch]).cuda()
    captions = longclip.tokenize([item['caption'] for item in batch], truncate=True).cuda()
    short_captions = longclip.tokenize([item['caption_short'] for item in batch], truncate=True).cuda()
    with torch.no_grad():
        features = model(images, captions, short_captions)
    features = [{key: value[i].cpu() for key, value in features.items()} for i in range(len(batch))]
    for item, feature in zip(batch, features):
        item['feature'] = feature
    return batch

def pre_featurize():
    model, preprocess = longclip.load(model_ckpt, device='cpu')
    model.cuda()

    dataset = ImageTextDataset(os.path.join(data4v_root, json_name), image_root)
    dataloader = data.DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=min(cpu_count(), 8), 
        pin_memory=True,
        collate_fn=lambda x: x  # Return list of items
    )

    json_new = []
    features = []

    for batch in tqdm(dataloader):
        processed_batch = process_batch(model, preprocess, batch)
        for item in processed_batch:
            json_new.append({
                'id': item['id'],
                'image': item['image_path'],
                'caption': item['caption'],
            })
            features.append(item['feature'])

    json_output = os.path.join(data4v_root, 'captions_400.json')
    features_output = os.path.join(data4v_root, 'features_400.pkl')

    with open(json_output, 'w') as fp:
        json.dump(json_new, fp, separators=(',', ':'))
    with open(features_output, 'wb') as fp:
        pickle.dump(features, fp)

if __name__ == '__main__':
    pre_featurize()
