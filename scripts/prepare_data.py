"""
Build stratified train/val/test split CSVs from the raw DermNet folder structure.

DermNet layout after download:
    data/raw/train/<ClassName>/<image>.jpg
    data/raw/test/<ClassName>/<image>.jpg   (optional — we re-split from train only)

Outputs written to data/processed/:
    train.csv, val.csv, test.csv  — columns: path, label, class_name
    classes.csv                   — index → class_name mapping

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/default.yaml --data_dir data/raw/train
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dermnet.config import load_config


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Walk folder-per-class structure → DataFrame with [path, label, class_name].

    Args:
        data_dir: Root directory where each sub-directory is a class name.

    Returns:
        DataFrame sorted by class_name for reproducibility.
    """
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"No sub-directories found in {data_dir}. "
                         "Expected one folder per class.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    records = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        images = list(class_dir.glob("*.[jJ][pP][gG]")) + \
                 list(class_dir.glob("*.[jJ][pP][eE][gG]")) + \
                 list(class_dir.glob("*.[pP][nN][gG]"))
        for img_path in images:
            records.append({
                "path": str(img_path),
                "label": class_to_idx[class_name],
                "class_name": class_name,
            })

    df = pd.DataFrame(records)
    print(f"\nFound {len(df):,} images across {len(class_names)} classes")
    print("\nClass distribution (sorted by count):")
    print(df.groupby("class_name").size().sort_values().to_string())
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split preserving class balance.

    Returns:
        (train_df, val_df, test_df)
    """
    # Step 1: carve out test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(df, df["label"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: carve out val from remaining
    adjusted_val = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val_df, train_val_df["label"]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test splits for DermNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None,
                        help="Override config data_dir (path to folder-per-class root)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir if args.data_dir else cfg.data.data_dir)

    if not data_dir.exists():
        print(f"\nERROR: data_dir not found: {data_dir}")
        print("Run: bash scripts/download_data.sh  first.")
        sys.exit(1)

    df = build_dataframe(data_dir)

    # Save class index mapping
    class_names = sorted(df["class_name"].unique())
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"class_name": class_names}).to_csv(out_dir / "classes.csv", index=True)

    train_df, val_df, test_df = split_dataframe(
        df,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.project.seed,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"\nSplits saved to {out_dir}/")
    print(f"  Train : {len(train_df):,} images")
    print(f"  Val   : {len(val_df):,} images")
    print(f"  Test  : {len(test_df):,} images")
    print(f"  Classes: {df['class_name'].nunique()}")
    print("\nNext step: python scripts/train.py  (or: make train)")


if __name__ == "__main__":
    main()
