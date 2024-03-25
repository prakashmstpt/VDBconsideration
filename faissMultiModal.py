import os
from PIL import Image
import numpy as np
os.environ['CURL_CA_BUNDLE'] = '/Users/sptpt/MSPref/Transformers/hfg.pem.1'
from sentence_transformers import SentenceTransformer
multimodal_model = SentenceTransformer('clip-ViT-B-32')
images = [
Image.open('myCats.jpeg'),
Image.open('myHorse2.jpeg'),
Image.open('myBird.jpeg'),
Image.open('myPlane.png'),
Image.open('myPlane2.png'),
Image.open('myPlane3.png'),
Image.open('myBike.png'),
Image.open('myBike2.png'),
Image.open('myBird2.png'),
Image.open('myHorse.jpeg')
]
imgembeddings = multimodal_model.encode(images)
sns = [
"My adorable cat",
"Horse in a Farm"
]
snsembeddings = multimodal_model.encode(sns)
print(imgembeddings.shape[0])
print(imgembeddings.shape[1])
print(snsembeddings.shape[0])
print(snsembeddings.shape[1])
embeddings = np.concatenate((imgembeddings,snsembeddings))
del multimodal_model 
import faiss
from faiss import write_index, read_index
import math
from faiss import index_factory
count = embeddings.shape[0]
dim = embeddings.shape[1]
storage = "Flat"
cells=min(round(math.sqrt(count)), int(count/39))
params = f"IVF{cells},{storage}"
index = index_factory(dim, params)
index.train(embeddings)
index.add(embeddings)
write_index(index, "mulModalIndex.bin")
print(index.ntotal)
