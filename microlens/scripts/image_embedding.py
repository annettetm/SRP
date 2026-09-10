from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms

BASE = Path("/home/mathew/SRP/microlens")
PROCESSED = BASE / "processed"
RAW_IMAGES = BASE / "data/microlens100k/MicroLens-100k_covers"
MODELS = BASE / "models"
EMB_DIR = BASE / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

item_df = pd.read_csv(PROCESSED / "item_df.csv")
item_df["item_id"] = item_df["item_id"].astype(str).str.strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
state = torch.load(MODELS / "resnet50_imagenet1k_v1.pth", map_location="cpu")
model.load_state_dict(state)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

embeddings = []
item_ids = []

with torch.no_grad():
    for item_id in item_df["item_id"].tolist():
        img_path = RAW_IMAGES / f"{item_id}.jpg"
        item_ids.append(item_id)
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            emb = model(x).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            emb = np.zeros(2048, dtype=np.float32)
        embeddings.append(emb)

image_embeddings = np.stack(embeddings, axis=0)
np.save(EMB_DIR / "image_embeddings.npy", image_embeddings)
np.save(EMB_DIR / "image_ids.npy", np.array(item_ids, dtype=object))
print(f"Saved {image_embeddings.shape[0]} embeddings")