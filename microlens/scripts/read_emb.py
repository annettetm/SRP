#import numpy as np

#arr = np.load("embeddings/text_embeddings.npy", allow_pickle=True)
#print(arr)
#print(arr.shape)
#print(arr.dtype)


import pickle
import numpy as np

path = "proxyrca_data/iid2ifeature.pkl"

with open(path, "rb") as f:
    d = pickle.load(f)

print(type(d))
print("num items:", len(d))

first_key = next(iter(d))
first_val = d[first_key]

print("first key:", first_key)
print("value type:", type(first_val))
print("vector length:", len(first_val))
print("first 10 values:", first_val[:10])

lengths = [len(v) for v in d.values()]
print("unique lengths:", sorted(set(lengths))[:10])
print("all same length:", len(set(lengths)) == 1)

arr = np.array(list(d.values()), dtype=np.float32)
print("array shape:", arr.shape)
print("finite:", np.isfinite(arr).all())