import json
from PIL import Image
import clip

import torch.utils.data as data
import os
import pickle

data4v_root = '../../Long-CLIP/ShareGPT4V/data/'  # 'sharegpt4v/data/'
json_name = 'captions.json'
tf_name = 'features.pkl'  # teacher_features


class share4v_kd_train_dataset(data.Dataset):
    def __init__(self):
        with open(os.path.join(data4v_root + json_name), 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)
        with open(os.path.join(data4v_root + tf_name), 'rb') as fp:
            self.tf_data = pickle.load(fp)

        _ , self.preprocess = clip.load("ViT-B/16")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['caption']
        caption = caption.replace("\n", " ")
        caption_short = caption.split(". ")[0]
        
        image_name = os.path.join(data4v_root + self.json_data[index]['image'])
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)

        teacher_feature = self.tf_data[self.json_data[index]['id']]

        return image_tensor, caption, caption_short, teacher_feature
