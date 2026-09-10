import numpy as np
embeddings = np.load("/home/mathew/SRP/Amazon_beauty/embeddings/image_embeddings.npy", allow_pickle=True).item()

asin, vec = next(iter(embeddings.items()))
print(asin)
print(vec.shape)

print(len(embeddings))
print(next(iter(embeddings.items())))