import os 
from transformers import pipeline
from sentence_transformers import SentenceTransformer
os.environ['CURL_CA_BUNDLE'] = '/Users/sptpt/MSPref/Transformers/hfg.pem.1'
model_embed = "all-MiniLM-L6-v2"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ARTICLE = open('forest.txt', 'r').read() 
docsum=summarizer(ARTICLE, max_length=256, min_length=30, do_sample=False)
modeli = SentenceTransformer(model_embed)
embeddings = modeli.encode(docsum)
print(docsum)
print(embeddings)

