import os
from PIL import Image
os.environ['CURL_CA_BUNDLE'] = '/Users/sptpt/MSPref/Transformers/hfg.pem.1'
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, CLIPModel
multiModal_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
img = Image.open('myCats.jpeg')
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=img, return_tensors="pt", padding=True
)
outputs = multiModal_model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print("Probability of cat and dog sentences to cat image embeddings {}".format(probs))
