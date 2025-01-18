from model import longclip
import mobileclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = ["checkpoints/longclip-B.pt", 'checkpoints/mobileclip_s0.pt'][0]
mobileclip_name = 'mobileclip_s0'

if 'mobile' in model_name:
    model, _, preprocess = mobileclip.create_model_and_transforms(mobileclip_name, pretrained=model_name, reparameterize=True, device=device)
    tokenizer = mobileclip.get_tokenizer(mobileclip_name)

else:
    model, preprocess = longclip.load(model_name, device=device)
    # model, preprocess = longclip.load_from_clip("ViT-B/16", device=device)
    tokenizer = longclip.tokenize

texts = [
    "The image depicts a clear, sunny day. A man is about to cross the street, appearing somewhat hurried. He is dressed in all black and carries a grayish-white backpack. The shadow cast on the ground suggests that the sun is behind him. Next to him, a red sedan is parked. On both sides of the street, there are several trees—some are lush and green, while others have already shed their leaves. In the distance, there is a tall red building.",
    "The image shows a serene forest path surrounded by towering trees. Sunlight filters through the leaves, casting dappled shadows on the soft dirt trail. A woman in a light blue jacket and hiking boots is walking along the path, carrying a small backpack. The air seems fresh, and the only sound is the rustling of the leaves in the breeze. Far off in the distance, a small wooden bench sits beside a babbling brook.",
    "The picture captures a lively city square just as the sun sets. People of all ages are walking, talking, and sitting at outdoor cafes. Streetlights are beginning to glow, adding a warm ambiance to the cool evening air. In the center, a large fountain is illuminated by colorful lights, and a man in a red jacket is taking a photo of the scene. Tall buildings surround the square, their windows reflecting the last rays of sunlight.",
    "The scene shows a breathtaking snowy mountain range, with jagged peaks rising sharply against a clear blue sky. A family of four, dressed in warm winter clothes, is skiing down a gentle slope. The snow is fresh and untouched in some areas, and a few snow-covered pine trees dot the landscape. In the distance, a wooden cabin with smoke rising from its chimney is nestled between the slopes, offering a cozy refuge from the cold.",
    "This image captures a bustling market street in an ancient city. The narrow cobblestone path is lined with small stalls selling colorful fabrics, fresh produce, and handmade pottery. A young man in a white shirt is bargaining with a vendor for a basket of fruit. The buildings around the market are made of old stone, with intricate carvings visible on their facades. The sky is clear, and the sound of chatter and laughter fills the air as people go about their shopping.",
    "The image shows a quiet beach at sunrise, with soft waves gently lapping at the shore. A lone figure, a woman in a wide-brimmed hat and flowing white dress, is walking barefoot along the water’s edge. The sky is painted in shades of orange and pink as the first light of the day begins to break through the horizon. In the background, a few seagulls are flying low over the water, while distant cliffs rise up against the sky.",
]

texts_hard = [
    "The image depicts a clear, sunny day. A man is about to cross the street, appearing somewhat hurried. He is dressed in all black and carries a grayish-white backpack. The shadow cast on the ground suggests that the sun is behind him. Next to him, a red sedan is parked. On both sides of the street, there are several trees—some are lush and green, while others have already shed their leaves. In the distance, there is a tall red building.",
    "The picture shows a bright afternoon under a clear sky. A woman in a light blue dress and a brown leather satchel is strolling across the street. A cyclist wearing a yellow helmet rides past her, and their shadows stretch long, hinting at the sun's low position in the sky. Along the roadside, there are flowering bushes interspersed with green trees. In the far background, a white office building with large glass windows catches the sunlight.",
    "A bustling street scene unfolds on a sunny morning. A child in a yellow raincoat runs ahead, dragging a small red wagon. A gray minivan drives past, and its shadow merges with that of a nearby lamppost. The sidewalks are lined with trees in full bloom, and a couple of benches are occupied by people chatting. In the background, a tall beige building with balconies stands prominently, its facade adorned with climbing ivy.",
    "This image captures a serene midday setting. A man dressed in a navy-blue jacket and jeans walks purposefully across the street, holding a cup of coffee. An orange motorcycle is parked near a hydrant, and the shadows on the pavement indicate that the sun is almost overhead. The street is flanked by modern-looking streetlights and neatly trimmed hedges. In the distance, a gray skyscraper with a rooftop garden dominates the skyline.",
    "The scene reveals a quiet street under a radiant sun. A woman in a red coat and carrying an umbrella crosses the road, her steps deliberate. Nearby, a black convertible with its roof down is parked under the shade of a tall, leafy tree. The sidewalk is bordered by a mix of trees—some green, others with bare branches. In the far background, a clock tower rises above a cluster of smaller buildings.",
    "The picture showcases a vibrant urban corner bathed in sunshine. A man wearing a beige hat and a plaid shirt steps off the curb, holding a briefcase. Beside him, a white delivery truck is stopped, with its rear doors open, revealing boxes inside. The street is lined with trees whose leaves are variegated shades of gold and orange, hinting at autumn. In the background, a dark blue building with a rooftop café adds a touch of charm to the scene.",
]

# text = longclip.tokenize(["A man is crossing the street with a red car parked nearby.", "A man is driving a car in an urban scene."]).to(device)
text = tokenizer(texts).to(device)  # texts_hard
image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

# print(model.logit_scale)

model.eval()

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (model.logit_scale.exp() * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
