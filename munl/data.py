"""Loading and preprocessing of the FNMA (Fannie Mae) loan-level dataset.

The task is regression: predict a loan's acquisition unpaid principal balance
(UPB) from borrower, property and census-tract attributes.

``Record Number`` is kept as the row identifier used to address unlearning
requests. It is *not* a model feature: it carries no predictive signal and
feeding it to the network (as the original notebook did) only adds noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "Record Number"
TARGET_COLUMN = "Acquisition Unpaid Principal Balance (UPB)"

NUMERIC_FEATURES = [
    "Borrower’s (or Borrowers’) Annual Income",
    "Area Median Family Income (2016)",
    "Tract Income Ratio",
    "2010 Census Tract - Median Income",
    "Number of Borrowers",
    "Rate Spread",
]

# Codes, not quantities: scaling e.g. a state code as a number is meaningless,
# so these are one-hot encoded instead.
CATEGORICAL_FEATURES = [
    "Purpose of Loan",
    "First-Time Home Buyer",
    "Federal Guarantee",
    "Property Type",
    "Occupancy Code",
    "US Postal State Code",
    "Metropolitan Statistical Area (MSA) Code",
    "Lien Status",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class Split:
    """One partition of the data, already transformed into model-ready arrays."""

    X: np.ndarray  # float32, shape (n, d)
    y: np.ndarray  # float32, shape (n,), standardized target
    ids: np.ndarray  # int64, shape (n,), record numbers


@dataclass
class PreparedData:
    train: Split
    test: Split
    preprocessor: ColumnTransformer
    target_scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return self.train.X.shape[1]

    def to_dollars(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map standardized targets/predictions back to the original UPB scale."""
        return self.target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def load_fnma(path: str) -> pd.DataFrame:
    """Read the FNMA loan CSV and keep only the columns the pipeline uses."""
    df = pd.read_csv(path)
    missing = [c for c in [ID_COLUMN, TARGET_COLUMN, *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df = df[[ID_COLUMN, *FEATURES, TARGET_COLUMN]].dropna(subset=[ID_COLUMN, TARGET_COLUMN])
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"'{ID_COLUMN}' must be unique to address unlearning requests")
    return df.reset_index(drop=True)


def make_synthetic(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Generate a dataset with the same schema as the FNMA file.

    Lets the whole pipeline (and the test suite) run without the real CSV.
    UPB is driven mostly by income and loan purpose, plus noise.
    """
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=11.2, sigma=0.5, size=n)
    area_income = rng.normal(70_000, 12_000, size=n).clip(30_000)
    purpose = rng.choice([1, 2, 6, 7, 8], size=n)
    borrowers = rng.choice([1, 2], size=n, p=[0.45, 0.55])
    df = pd.DataFrame(
        {
            ID_COLUMN: rng.permutation(np.arange(1, n + 1) * 7 + 1000),
            "Borrower’s (or Borrowers’) Annual Income": income,
            "Area Median Family Income (2016)": area_income,
            "Tract Income Ratio": rng.uniform(0.4, 2.0, size=n),
            "2010 Census Tract - Median Income": rng.normal(65_000, 15_000, size=n).clip(20_000),
            "Number of Borrowers": borrowers,
            "Rate Spread": rng.choice([0.0, 0.0, 0.0, 1.5, 2.0], size=n),
            "Purpose of Loan": purpose,
            "First-Time Home Buyer": rng.choice([1, 2], size=n),
            "Federal Guarantee": rng.choice([1, 2, 3, 4], size=n),
            "Property Type": rng.choice(["PT01", "PT02", "PT07"], size=n),
            "Occupancy Code": rng.choice([1, 2, 3], size=n),
            "US Postal State Code": rng.choice(np.arange(1, 57), size=n),
            "Metropolitan Statistical Area (MSA) Code": rng.choice(np.arange(10_000, 10_050), size=n),
            "Lien Status": rng.choice([1, 2], size=n, p=[0.95, 0.05]),
        }
    )
    df[TARGET_COLUMN] = (
        2.8 * income
        + 0.5 * area_income
        + 15_000 * (borrowers - 1)
        - 40_000 * (purpose == 8)
        + rng.normal(0, 25_000, size=n)
    ).clip(10_000)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def prepare(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> PreparedData:
    """Split into train/test and fit all transforms on the training split only."""
    df = df.copy()
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    preprocessor = build_preprocessor()
    target_scaler = StandardScaler()
    X_train = preprocessor.fit_transform(train_df[FEATURES])
    y_train = target_scaler.fit_transform(train_df[[TARGET_COLUMN]]).ravel()
    X_test = preprocessor.transform(test_df[FEATURES])
    y_test = target_scaler.transform(test_df[[TARGET_COLUMN]]).ravel()

    def split(X, y, frame) -> Split:
        return Split(
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(y, dtype=np.float32),
            ids=frame[ID_COLUMN].to_numpy(dtype=np.int64),
        )

    return PreparedData(
        train=split(X_train, y_train, train_df),
        test=split(X_test, y_test, test_df),
        preprocessor=preprocessor,
        target_scaler=target_scaler,
    )
