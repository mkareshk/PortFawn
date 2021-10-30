import json
import hashlib
from pathlib import Path

import numpy as np


def get_asset_hash(asset_list):
    return hashlib.md5(".".join(asset_list).encode("utf-8")).hexdigest()[0:5]


def get_assets_signature(asset_list, start, end):
    asset_hash = get_asset_hash(asset_list)
    return Path(f"{start}___{end}___{asset_hash}")


def get_freq_list():
    return {"D": "daily", "M": "monthly", "Q": "quarterly", "Y": "annual"}


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def remove_one_asset_portfolio(df, asset_list):
    df = df.loc[:, ~df.columns.duplicated()]
    for c in df.columns:
        if c.split()[0].upper() in asset_list and c.count("Portfolio") > 0:
            df.drop(c, axis=1, inplace=True)
    return df
