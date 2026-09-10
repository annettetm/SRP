import argparse
import pickle
import shutil

from datetime import datetime as dt
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_NEGATIVE_SAMPLES = 100
USE_FILTER_OUT = False
MIN_ITEM_COUNT_PER_USER = 5
MIN_USER_COUNT_PER_ITEM = 5
ICONTEXT_COLUMNS = ["year", "month", "day", "dayofweek", "dayofyear", "week"]

BASE = Path("/home/mathew/SRP/ProxyRCA1")
PROCESSED = BASE / "processed/beauty1"
EMB = BASE / "embeddings/beauty1"
DATA_DIR = BASE / "data/beauty1"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def print_timedelta(tdo):
    print(f"({'.'.join(str(tdo).split('.')[:-1])})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--random_seed", type=int, default=12345)
    return p.parse_args()


def append_icontext(df_rows):
    df_rows = df_rows.copy()
    df_rows["dto"] = pd.to_datetime(df_rows["stamp"], unit="ms")
    df_rows["year"] = df_rows["dto"].dt.year
    df_rows["month"] = df_rows["dto"].dt.month
    df_rows["day"] = df_rows["dto"].dt.day
    df_rows["dayofweek"] = df_rows["dto"].dt.dayofweek
    df_rows["dayofyear"] = df_rows["dto"].dt.dayofyear
    df_rows["week"] = df_rows["dto"].dt.isocalendar().week.astype(int)
    df_rows["year"] = df_rows["year"] - df_rows["year"].min()
    ymax = df_rows["year"].max()
    df_rows["year"] = df_rows["year"] / ymax if ymax != 0 else 0.0
    df_rows["year"] = df_rows["year"].fillna(0.0)
    df_rows["month"] /= 12
    df_rows["day"] /= 31
    df_rows["dayofweek"] /= 7
    df_rows["dayofyear"] /= 365
    df_rows["week"] /= 52
    df_rows = df_rows.drop(columns=["dto"])
    return df_rows[["uid", "iid", "stamp"] + ICONTEXT_COLUMNS]


def load_or_build_iid2ifeature():
    pkl = DATA_DIR / "iid2ifeature.pkl"
    if pkl.is_file():
        with open(pkl, "rb") as f:
            return pickle.load(f)

    item_df = pd.read_csv(PROCESSED / "item_df.csv")
    item_df["parent_asin"] = item_df["parent_asin"].astype(str).str.strip()

    img_asins = np.load(EMB / "image_asins.npy", allow_pickle=True)
    txt_asins = np.load(EMB / "text_asins.npy", allow_pickle=True)
    img_emb = np.load(EMB / "image_embeddings.npy")
    txt_emb = np.load(EMB / "text_embeddings.npy")

    img_map = {str(a).strip(): img_emb[i] for i, a in enumerate(img_asins)}
    txt_map = {str(a).strip(): txt_emb[i] for i, a in enumerate(txt_asins)}

    dim = img_emb.shape[1] + txt_emb.shape[1]
    iid2ifeature = {}
    for iid, asin in enumerate(item_df["parent_asin"].tolist(), start=1):
        if asin in img_map and asin in txt_map:
            feat = np.concatenate([img_map[asin], txt_map[asin]]).astype(np.float32)
        elif asin in img_map:
            feat = img_map[asin].astype(np.float32)
        elif asin in txt_map:
            feat = txt_map[asin].astype(np.float32)
        else:
            feat = np.zeros(dim, dtype=np.float32)
        iid2ifeature[iid] = tuple(feat.tolist())

    with open(pkl, "wb") as f:
        pickle.dump(iid2ifeature, f)

    return iid2ifeature



def build_and_export_image_text_features():
    data_dir = DATA_DIR
    # load raw embeddings
    img_asins = np.load(EMB / 'image_asins.npy', allow_pickle=True)
    txt_asins = np.load(EMB / 'text_asins.npy', allow_pickle=True)
    img_emb = np.load(EMB / 'image_embeddings.npy', allow_pickle=True)
    txt_emb = np.load(EMB / 'text_embeddings.npy', allow_pickle=True)

    # normalize asin keys to str
    img_map = {str(a).strip(): img_emb[i].astype(np.float32) for i, a in enumerate(img_asins)}
    txt_map = {str(a).strip(): txt_emb[i].astype(np.float32) for i, a in enumerate(txt_asins)}

    # load item list (processed/item_df.csv) and parent_asin column
    item_df = pd.read_csv(PROCESSED / 'item_df.csv')
    parent_asins = item_df['parent_asin'].astype(str).str.strip().tolist()

    # build iid->image/text dicts (iid starts at 1)
    iid2image = {}
    iid2text = {}
    img_dim = img_emb.shape[1] if img_emb.ndim > 1 else 1
    txt_dim = txt_emb.shape[1] if txt_emb.ndim > 1 else 1
    img_zero = np.zeros(img_dim, dtype=np.float32)
    txt_zero = np.zeros(txt_dim, dtype=np.float32)

    for iid, asin in enumerate(parent_asins, start=1):
        iid2image[iid] = tuple(img_map.get(asin, img_zero).tolist())
        iid2text[iid] = tuple(txt_map.get(asin, txt_zero).tolist())

    # save per-iid pickles in Amazon_beauty proxyrca_data
    with open(data_dir / 'iid2imagefeature.pkl', 'wb') as fp:
        pickle.dump(iid2image, fp)
    with open(data_dir / 'iid2textfeature.pkl', 'wb') as fp:
        pickle.dump(iid2text, fp)

    # For backward-compatibility with existing preprocess that expects 'ifeatures.pkl' as image features:
    #with open(data_dir / 'ifeatures.pkl', 'wb') as fp:
     #   pickle.dump(imagefeatures, fp)

    return iid2image, iid2text

    # Copy results into the ProxyRCA1 repo data folder (dname)
   # repo_dir = Path(repo_data_root) / dname
    #repo_dir.mkdir(parents=True, exist_ok=True)

    #shutil.copyfile(data_dir / 'iid2imagefeature.pkl', repo_dir / 'iid2imagefeature.pkl')
    #shutil.copyfile(data_dir / 'iid2textfeature.pkl', repo_dir / 'iid2textfeature.pkl')
    #shutil.copyfile(data_dir / 'imagefeatures.pkl', repo_dir / 'ifeatures.pkl')      # image-only -> ifeatures.pkl
    #shutil.copyfile(data_dir / 'textfeatures.pkl', repo_dir / 'textfeatures.pkl')

    #print(f"Exported image/text features to {repo_dir}")





def do_general_preprocessing(args, df_rows):
    data_dir = DATA_DIR

    if USE_FILTER_OUT:
        df_iid2ucount = df_rows.groupby("iid").size()
        survived_iids = df_iid2ucount.index[df_iid2ucount >= MIN_USER_COUNT_PER_ITEM]
        df_rows = df_rows[df_rows["iid"].isin(survived_iids)]

        df_uid2icount = df_rows.groupby("uid").size()
        survived_uids = df_uid2icount.index[df_uid2icount >= MIN_ITEM_COUNT_PER_USER]
        df_rows = df_rows[df_rows["uid"].isin(survived_uids)]

    check = dt.now()
    ss_uids = df_rows.groupby("uid").size().sort_values(ascending=False)
    uid2uindex = {uid: index for index, uid in enumerate(list(ss_uids.index), start=1)}
    df_rows["uindex"] = df_rows["uid"].map(uid2uindex)
    df_rows = df_rows.drop(columns=["uid"])
    with open(data_dir / "uid2uindex.pkl", "wb") as fp:
        pickle.dump(uid2uindex, fp)
    print("- map uid -> uindex", end=" ")
    print_timedelta(dt.now() - check)

    check = dt.now()
    ss_iids = df_rows.groupby("iid").size().sort_values(ascending=False)
    iid2iindex = {iid: index for index, iid in enumerate(list(ss_iids.index), start=1)}
    df_rows["iindex"] = df_rows["iid"].map(iid2iindex)
    df_rows = df_rows.drop(columns=["iid"])
    with open(data_dir / "iid2iindex.pkl", "wb") as fp:
        pickle.dump(iid2iindex, fp)
    print("- map iid -> iindex", end=" ")
    print_timedelta(dt.now() - check)

    check = dt.now()
    df_rows["icontext"] = df_rows[ICONTEXT_COLUMNS].apply(tuple, axis=1)
    df_rows = df_rows.drop(columns=ICONTEXT_COLUMNS)
    df_rows = df_rows[["uindex", "iindex", "stamp", "icontext"]]
    df_rows.to_pickle(data_dir / "df_rows.pkl")
    print("- save df_rows with icontext", end=" ")
    print_timedelta(dt.now() - check)

    check = dt.now()
    uindex2urows_train = {}
    uindex2urows_valid = {}
    uindex2urows_test = {}
    for uindex in tqdm(list(uid2uindex.values()), desc="* splitting"):
        df_urows = df_rows[df_rows["uindex"] == uindex]
        urows = list(df_urows[["iindex", "stamp", "icontext"]].itertuples(index=False, name=None))
        if len(urows) < 3:
            uindex2urows_train[uindex] = urows
        else:
            uindex2urows_train[uindex] = urows[:-2]
            uindex2urows_valid[uindex] = urows[-2:-1]
            uindex2urows_test[uindex] = urows[-1:]

    with open(data_dir / "uindex2urows_train.pkl", "wb") as fp:
        pickle.dump(uindex2urows_train, fp)
    with open(data_dir / "uindex2urows_valid.pkl", "wb") as fp:
        pickle.dump(uindex2urows_valid, fp)
    with open(data_dir / "uindex2urows_test.pkl", "wb") as fp:
        pickle.dump(uindex2urows_test, fp)
    print("- save splits", end=" ")
    print_timedelta(dt.now() - check)

    return df_rows


def do_general_random_negative_sampling(args, df_rows):
    data_dir = DATA_DIR
    with open(data_dir / "uid2uindex.pkl", "rb") as fp:
        uid2uindex = pickle.load(fp)
    with open(data_dir / "iid2iindex.pkl", "rb") as fp:
        iid2iindex = pickle.load(fp)

    num_users = len(uid2uindex)
    num_items = len(iid2iindex)
    ns = {}
    rng = Random(args.random_seed)

    for uindex in tqdm(range(1, num_users + 1), desc="* sampling"):
        seen = set(df_rows[df_rows["uindex"] == uindex]["iindex"])
        sampled = set()
        while len(sampled) < NUM_NEGATIVE_SAMPLES:
            iindex = rng.randint(1, num_items)
            if iindex in seen or iindex in sampled:
                continue
            sampled.add(iindex)
        ns[uindex] = list(sampled)

    with open(data_dir / "ns_random.pkl", "wb") as fp:
        pickle.dump(ns, fp)


def do_create_ifeature_matrix():
    data_dir = DATA_DIR
    with open(data_dir / "iid2iindex.pkl", "rb") as fp:
        iid2iindex = pickle.load(fp)
    with open(data_dir / "iid2ifeature.pkl", "rb") as fp:
        iid2ifeature = pickle.load(fp)

    iindex2iid = {iindex: iid for iid, iindex in iid2iindex.items()}
    ifeatures = []
    for iindex in range(1, len(iid2iindex) + 1):
        iid = iindex2iid[iindex]
        ifeatures.append(iid2ifeature[iid])

    ifeature_dim = len(ifeatures[0])
    ifeatures = np.array([np.zeros(ifeature_dim, dtype=np.float32)] + ifeatures, dtype=np.float32)

    with open(data_dir / "ifeatures.pkl", "wb") as fp:
        pickle.dump(ifeatures, fp)


def main():
    args = parse_args()
    load_or_build_iid2ifeature()
    build_and_export_image_text_features()

    interaction_path = PROCESSED / "interaction_df.csv"
    df = pd.read_csv(interaction_path)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "user_id":
            rename_map[c] = "uid"
        elif cl == "item_idx":
            rename_map[c] = "iid"
        elif cl == "timestamp":
            rename_map[c] = "stamp"

    df = df.rename(columns=rename_map)

    if not {"uid", "iid", "stamp"}.issubset(df.columns):
        raise ValueError(f"interaction_df.csv must contain uid/iid/stamp after renaming. Found: {list(df.columns)}")

    df = df[["uid", "iid", "stamp"]].copy()
    df["uid"] = df["uid"].astype(str).str.strip()
    df["iid"] = df["iid"].astype(int)+1

    df = df.sort_values("stamp", ascending=True).reset_index(drop=True)
    df = append_icontext(df)

    df_rows = do_general_preprocessing(args, df)
    do_general_random_negative_sampling(args, df_rows)
    do_create_ifeature_matrix()

    print("done")


if __name__ == "__main__":
    main()