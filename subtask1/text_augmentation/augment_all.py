# scripts/augment_all.py

import argparse
import os
import pandas as pd

from text_augment import augment_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Base data directory (containing raw/ and processed/)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=10_000,
        help="Target number of examples per language",
    )
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "subtask1", "train")
    out_dir = os.path.join(args.data_dir, "subtask1_jumbo")
    os.makedirs(out_dir, exist_ok=True)

    lang_files = [
        ("eng", "eng.csv"),
        ("spa", "spa.csv"),
        ("zho", "zho.csv"),
        ("deu", "deu.csv"),
        ("arb", "arb.csv"),
    ]

    for lang, fname in lang_files:
        in_path = os.path.join(raw_dir, fname)
        if not os.path.exists(in_path):
            print(f"[WARN] Missing file for {lang}: {in_path}, skipping.")
            continue

        print(f"[INFO] Reading {in_path}")
        df = pd.read_csv(in_path)

        # This will add/overwrite the 'language' column with the correct lang code
        full_df = augment_df(df, lang, args.target_size)

        out_path = os.path.join(
            out_dir, f"{lang}_{args.target_size}.csv"
        )
        full_df.to_csv(out_path, index=False)
        print(
            f"[INFO] {lang}: {full_df.shape[0]} rows "
            f"with columns {list(full_df.columns)} -> {out_path}"
        )


if __name__ == "__main__":
    main()
