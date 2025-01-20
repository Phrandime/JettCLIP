import torch
import sys
sys.path.append('../../..')
from model import longclip
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
# from templates import imagenet_templates

from tqdm import tqdm
import mobileclip

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = longclip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

model_name = "/path/to/jett_s0.pt"

model_type = ['longCLIP', 'mobileclip_s0'][0]
device = "cuda" if torch.cuda.is_available() else "cpu"

if model_type == "longCLIP":
    model, preprocess = longclip.load(model_name, device=device)
    tokenizer = longclip.tokenize
elif model_type[:10] == "mobileclip":
    model, _, preprocess = mobileclip.create_model_and_transforms(model_type, pretrained=model_name, reparameterize=True, device=device)
    tokenizer = mobileclip.get_tokenizer(model_type)

model.eval()

testset = torchvision.datasets.CIFAR100(root="data/cifar100", train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

text_feature = zeroshot_classifier(model, testset.classes, imagenet_templates)

correct = 0
total = 0
with torch.no_grad():
    i = 0
    for data in testset:
        images, labels = data
        images = preprocess(images).unsqueeze(0).to(device)

        image_feature = model.encode_image(images)
        image_feature = image_feature/image_feature.norm(dim=-1, keepdim=True)
        
        sims = image_feature @ text_feature
        
        pred = torch.argmax(sims, dim=1).item()
        if labels == pred:
            correct += 1
        total += 1

    print(correct)
    print(total)
    print(correct/total)

print("Accuracy of the CLIP model on the CIFAR-100 test images: %d %%" % (100 * correct / total))