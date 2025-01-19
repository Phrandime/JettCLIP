# Jett-CLIP

"Multimodal Learning" Course Project

The name "Jett" refer to cartoon Super Wings, where the main character is Jett, and Chinese name is "乐迪".

**Jett-CLIP: Faster MobileCLIP in Long Text-Image Alignment**\
*Yangdi Yue, Xiaole Wang*

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
    
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 

```

### Evaluation
Please find the detailed evaluation results [here](./results).

To reproduce results, please download our pretrained checkpoints, please download in [JettCLIP](www.baidu.com) and put it in `checkpoints/`. We provide code to perform zero-shot classification evaluation on cifar-10/cifar-100 dataset and retrieval evaluation on Urban1k dataset, please refer to `eval/`.


