import os
os.environ['CURL_CA_BUNDLE'] = '/Users/sptpt/MSPref/Transformers/hfg.pem.1'
from sentence_transformers import SentenceTransformer
models = [
"all-MiniLM-L6-v2",
"hkunlp/instructor-base",
"all-mpnet-base-v2",
"mixedbread-ai/mxbai-embed-large-v1",
]
sns="How Embedings varies with model?"
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
print("Model Name"+"\t\t"+"Length of Embeddings"+"\t\t"+"Embeddings")
for model in models:
	modeli = SentenceTransformer(model)
	embeddings = modeli.encode(sns)
	lenEmbeds = len(embeddings)
	embeds = formatEmbedsing(embeddings)
	#print(embeds)
	if "/" in model:
		model = model.split("/")[1]
	print(model,"\t\t",lenEmbeds,"\t",embeds)
