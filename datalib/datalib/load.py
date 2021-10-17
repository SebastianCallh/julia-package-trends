from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional, Union
from glob import glob

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


@dataclass
class RequestCountProblem:
    dates: np.ndarray
    request_counts_log10: np.ndarray
    regions: np.ndarray
    client_types: np.ndarray
    region_to_index: Dict[str, int]
    client_type_to_index: Dict[str, int]
    feature_scaler: StandardScaler


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
        .rename(
            columns={
                "date": "Date",
                "region": "Region",
                "client_type": "Client type",
                "request_count": "Request count",
            }
        )
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Request count (log10)"] = np.log10(df["Request count"])
    return df


def construct_problem(df: pd.DataFrame) -> RequestCountProblem:
    scaler = StandardScaler()
    regions = df["Region"].values
    client_types = df["Client type"].values
    dates = scaler.fit_transform(df["Date"].values.reshape(-1, 1)).flatten()
    return RequestCountProblem(
        dates=dates,
        request_counts_log10=df["Request count (log10)"].values,
        regions=regions,
        client_types=client_types,
        region_to_index={r: i for i, r in enumerate(np.unique(regions))},
        client_type_to_index={c: i for i, c in enumerate(np.unique(client_types))},
        feature_scaler=scaler,
    )
