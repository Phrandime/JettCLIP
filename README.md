# Jett-CLIP

**"Multimodal Learning" Course Project —— Jett-CLIP: Faster MobileCLIP in Long Text-Image Alignment**\
*Yangdi Yue, Xiaole Wang*

The name "Jett" refer to cartoon Super Wings, where the main character is Jett, and Chinese name is "乐迪".

- **Update 2025/1/20:** Releasing our code.

### Highlights
* `JettCLIP` is a variant of `MobileCLIP` and `LongCLIP`, which obtains better long text ability than MobileCLIP and is faster than LongCLIP.

## Getting Started

### Setup

Create conda environment
```bash
git clone https://github.com/Phrandime/JettCLIP.git
cd JettCLIP
conda create -n clipenv python=3.10
conda activate clipenv
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:/path/to/Jett-CLIP

```

Installing mobileclip is recommended
```bash
git clone https://github.com/apple/ml-mobileclip.git
cd ml-mobileclip
pip install -e .
```

### Usage Example
To models from the official repo, follow the code snippet below
```python
from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("/path/to/jett_s0.pt", device=device)
text = longclip.tokenize(["A man is crossing the street with a red car parked nearby.", "A man is driving a car in an urban scene."]).to(device)
image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    logits_per_image = model.logit_scale.exp()* image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 

```

### Evaluation
Please find the detailed evaluation results in our report.

To reproduce results, please download our pretrained checkpoints in [JettCLIP](https://disk.pku.edu.cn/link/AA49DEF1014F764F29A11DB4E4EB158953) and put it in `checkpoints/`. We provide code to perform zero-shot classification evaluation on cifar-10/cifar-100 dataset and retrieval evaluation on COCO(download from [COCO](https://cocodataset.org/#download)) and Urban1k dataset(download from [Urban1k](https://huggingface.co/datasets/BeichenZhang/Urban1k/resolve/main/Urban1k.zip)), please refer to `eval/`.

Organize the data as follows in `/ShareGPT4V/data`:
```none
ShareGPT4V
├── ...
├── data
│   ├── Urban1k
│   │   ├── caption
│   │   ├── image
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
├── ...
```

After getting data ready, run:
```bash
# pay attention to checkpoint path
python -m eval.retrieval.Urban1k # for long text retrieval
python -m eval.retrieval.coco # for short text retrieval
python -m eval.classification.cifar10 
python -m eval.classification.cifar100 # for classification
```

### Train

To run the training code for Jett-CLIP, please follow the following step, take mobileclip-s0 for example.

#### 1. Prepare MobileCLIP Model
First, download the checkpoints of MobileCLIP-s0. You can refer to this page https://github.com/apple/ml-mobileclip.

Then, you can load the model from MobileCLIP by running the following command. The positional embedding will be stretched from 77 to 248. 
```python
from model import longclip
model_name = "/path/to/mobileclip-s0.pt"
model, preprocess = longclip.load_from_clip(model_name, device='cpu')
```
*Note: Due to the different usage of positional encoding in MobileCLIP compared to this model, it is normal to observe degraded performance after loading the model parameters.*

#### 2. Prepare COCO and LLAVA dataset

First, download all images we used.
- LAION-CC-SBU-558K: [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)

Then, download the long caption and teacher embedding of these image [features](https://disk.pku.edu.cn/link/AAA9F2A82E46834E8B9DE5711C9260C1CF)

Finally, organize the data as follows in `/ShareGPT4V/data`:

```none
ShareGPT4V
├── ...
├── data
|   ├── captions.json
|   ├── features.pkl
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
├── ...
```

#### 3. Finetune

Finally, you can run the `train_kd.py` for fine-tuning. Or start it from train.sh. The code will store every 500 steps, and you can also find the performance improvement through tensorboard logfile at `longclip/`.

```bash
bash train_jettclip/train.sh
```

If you want to train from stratch, please run:
```bash
python train_jettclip/train_kd.py --base_model "jett" # not recommend
```
