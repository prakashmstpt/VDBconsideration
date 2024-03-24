import os
from PIL import Image
import numpy as np
os.environ['CURL_CA_BUNDLE'] = '/Users/smtpt/MSPref/Transformers/hfg.pem.1'
from sentence_transformers import SentenceTransformer
def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    denominator = a_norm * b_norm
    
    cosine_similarity = nominator / denominator
    
    return cosine_similarity
def formatEmbedsing(embedding):
        if len(embedding) > 6:
                listStr=""
                listStr = ' '.join(map(str,embedding[0:3]))
                listStr = listStr + " .. "
                listStre = ' '.join(map(str,embedding[-2:]))
                listStr = listStr + listStre
                return listStr
        else:
                return embedding
multimodal_model = SentenceTransformer('clip-ViT-B-32')
images = [
Image.open('elephantjpg.jpeg'),
Image.open('elephantpng.png'),
Image.open('ElephantAreal.png'),
Image.open('a-rare-white-elephant.jpg')
]
imgembeddings = multimodal_model.encode(images)
#calculating cosine of same image different formats
ejpg=imgembeddings[0]
#print(ejpg)
epng=imgembeddings[1]
# Calculate cosine similarity
similarity = cosine_similarity(ejpg, epng)
print("JPG vector embdings with a length {} --  {}".format(len(ejpg),formatEmbedsing(ejpg)))
print("PNGvector embdings with a length {} --  {}".format(len(epng),formatEmbedsing(epng)))
print("Similarity of same image in JPG vs PNG {}".format(similarity))
eareal=imgembeddings[2]
similarity = cosine_similarity(ejpg, eareal)
print("OBJ vector embdings with a length {} --  {}".format(len(ejpg),formatEmbedsing(ejpg)))
print("OBJ Areal vector embdings with a length {} --  {}".format(len(eareal),formatEmbedsing(eareal)))
print("Similarity of same object differnt orientation {}".format(similarity))
ewrare=imgembeddings[3]
similarity = cosine_similarity(ejpg, ewrare)
print("OBJ vector embdings with a length {} --  {}".format(len(ejpg),formatEmbedsing(ejpg)))
print("OBJ color vector embdings with a length {} --  {}".format(len(ewrare),formatEmbedsing(ewrare)))
print("Similarity of same object differnt color {}".format(similarity))
