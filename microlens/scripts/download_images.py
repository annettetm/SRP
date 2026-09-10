import pandas as pd
import ast
import requests
from io import BytesIO
from PIL import Image
import os

meta_df = pd.read_csv("home/mathew/SRP/Amazon_beauty/processed/preprocessed_meta.csv")


def extract_image_url(x):
    if pd.isna(x):
        return None

    try:
        obj = ast.literal_eval(x) if isinstance(x, str) else x

        # CASE 1: list of images
        if isinstance(obj, list) and len(obj) > 0:
            first_img = obj[0]

            if isinstance(first_img, dict):
                return (
                    first_img.get("hi_res")
                    or first_img.get("large")
                    or first_img.get("thumb")
                )

        # CASE 2: dict directly
        if isinstance(obj, dict):
            return (
                obj.get("hi_res")
                or obj.get("large")
                or obj.get("thumb")
            )

    except Exception:
        return None

    return None

meta_df["image_url"] = meta_df["images"].apply(extract_image_url)
print("Valid URLs:", meta_df["image_url"].notna().sum())
print(meta_df["image_url"].dropna().head(5))

out_dir = "/home/mathew/SRP/Amazon_beauty/images"
os.makedirs(out_dir, exist_ok=True)

saved = 0
for _, row in meta_df.iterrows():
    url = row["image_url"]
    asin = row["parent_asin"]

    if not isinstance(url, str) or not url.strip():
        continue

    try:
        print(f"Downloading: {asin}")

        r = requests.get(url, timeout=20)

        print("Status:", r.status_code)

        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")

        save_path = os.path.join(out_dir, f"{asin}.jpg")

        img.save(save_path)

        print("Saved to:", save_path)

        saved += 1
    except Exception:
        print("FAILED:", asin)
        print("ERROR:", e)



print("Saved:", saved)