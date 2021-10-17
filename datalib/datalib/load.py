import os
from pathlib import Path
from typing import Optional, Union
from glob import glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
ALLOWED_REGIONS = set(
    (
        "es-east",
        "us-central",
        "us-west",
        "eu-central",
        "cn-east",
        "cn-northeast",
    )
)


def root_path(path: Optional[Union[Path, str]] = None) -> Path:
    def go(p):
        if glob(str(p) + "/.gitignore"):
            return p
        else:
            return go(p.parent)

    return go(Path(path or os.getcwd()))


def requests_by_region_by_date() -> pd.DataFrame:
    df = (
        pd.read_csv(root_path() / DATA_DIR / "package_requests_by_region_by_date.csv")
        .query("region in @ALLOWED_REGIONS")
        .dropna(subset=["client_type"])
        .filter(["date", "region", "client_type", "request_count"])
        .groupby(["date", "region", "client_type"])
        .sum()
        .reset_index()
        .rename(columns={
            "date": "Date",
            "region": "Region",
            "client_type": "Client type",
            "request_count": "Request count"
        })
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Request count (log10)"] =  np.log10(df["Request count"])
    return df
