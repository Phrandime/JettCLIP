import sys
sys.path.append('../..')
from model import longclip
import torch
from torchvision.datasets import CocoCaptions
from PIL import Image
import mobileclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/path/to/jett_s0.pt"
model_type = ['longCLIP', 'mobileclip_s0'][0]

if model_type == "longCLIP":
    model, preprocess = longclip.load(model_name, device=device)
    tokenizer = longclip.tokenize
elif model_type[:10] == "mobileclip":
    model, _, preprocess = mobileclip.create_model_and_transforms(model_type, pretrained=model_name, reparameterize=True, device=device)
    tokenizer = mobileclip.get_tokenizer(model_type)

model.eval()

coco = CocoCaptions(root="ShareGPT4V/data/coco/val2017/", annFile="ShareGPT4V/data/coco/annotations/captions_val2017.json", transform=None)

image_features = []
text_features = []
pred_true = 0

with torch.no_grad():
    for image, captions in coco:
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features.append(model.encode_image(image_input))

        captions = captions[0:5]
        caption_input = longclip.tokenize(captions).to(device)
        text_features.extend(model.encode_text(caption_input))

    image_features = torch.stack(image_features).squeeze()
    image_features /= image_features.norm(dim=-1, keepdim=True)

    print(image_features.shape)
    text_features = torch.stack(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = image_features.squeeze() @ text_features.squeeze().T
   
    print("I2T")
    for i in range(5000):
        pred = similarity[i]
        b = pred.argsort()[-1:]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in b:
                pred_true = pred_true + 1
                break
    print(pred_true / 5000)
    pred_true = 0

    for i in range(5000):
        pred = similarity[i]
        b = pred.argsort()[-5:]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in b:
                pred_true = pred_true + 1
                break
    print(pred_true / 5000)
    pred_true = 0

    for i in range(5000):
        pred = similarity[i]
        b = pred.argsort()[-10:]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in b:
                pred_true = pred_true + 1
                break
    print(pred_true / 5000)
    pred_true = 0

    print("T2I")
    similarity = similarity.T
    for i in range(25000):
        pred = similarity[i]
        b = pred.argsort()[-1:]
        true_index = i//5
        if true_index in b:
            pred_true = pred_true + 1

    print(pred_true/25000)
    pred_true = 0

    for i in range(25000):
        pred = similarity[i]
        b = pred.argsort()[-5:]
        true_index = i//5
        if true_index in b:
            pred_true = pred_true + 1

    print(pred_true/25000)
    pred_true = 0

    for i in range(25000):
        pred = similarity[i]
        b = pred.argsort()[-10:]
        true_index = i//5
        if true_index in b:
            pred_true = pred_true + 1

    print(pred_true/25000)
    