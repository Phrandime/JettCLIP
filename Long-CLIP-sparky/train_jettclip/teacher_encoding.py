import json
from PIL import Image
import torch
import torch.utils.data as data
import os
from tqdm import tqdm
import pickle

import sys
sys.path.append("..")

from model import longclip


data4v_root = '../../Long-CLIP/ShareGPT4V/data/'  # 'sharegpt4v/data/'
json_name = 'share-captioner_coco_lcs_676k_1107.json'
image_root = '../../Long-CLIP/ShareGPT4V/data/'  # 'sharegpt4v/data/'

model_ckpt = '../checkpoints/longclip-B.pt'
assert torch.cuda.is_available()


def pre_featurize():
    model, preprocess = longclip.load(model_ckpt, device='cuda')

    # model, preprocess = longclip.load(model_ckpt, device='cpu')
    # model.cuda()

    with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    
    json_new = []
    features = []
    for index, data in enumerate(tqdm(json_data)):
        caption = json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        caption_short = caption.split(". ")[0]
        image_name = image_root + json_data[index]['image']
        image = Image.open(image_name)

        text = longclip.tokenize(caption, truncate=True).cuda()
        short_text = longclip.tokenize(caption_short, truncate=True).cuda()
        image_tensor = preprocess(image).unsqueeze(0).cuda()

        with torch.no_grad():
            feature = model(image_tensor, text, short_text)  # a dict
        
        for key, val in feature.items():
            print(key, val.dtype)
        raise

        feature = {key: value.cpu() for key, value in feature.items()}
        features.append(feature)

        json_new.append({
            'id': index,
            'image': json_data[index]['image'],
            'caption': json_data[index]['conversations'][1]['value'],
        })
    
    json_output = os.path.join(data4v_root, 'captions.json')
    features_output = os.path.join(data4v_root, 'features.pkl')

    with open(json_output, 'w') as fp:
        json.dump(json_new, fp, separators=(',', ':'))
    with open(features_output, 'wb') as fp:
        pickle.dump(features, fp)


if __name__ == '__main__':
    pre_featurize()