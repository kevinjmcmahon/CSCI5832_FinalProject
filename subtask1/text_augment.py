# src/augmentation/text_augment.py

import pandas as pd
import numpy as np
import random
from math import ceil

# For reproducibility (optional)
random.seed(42)
np.random.seed(42)

# Language-specific neutral-ish prefixes and suffixes
LANG_PREFIXES = {
    "eng": [
        "Honestly, ",
        "Tbh, ",
        "In my opinion, ",
        "Let’s be real: ",
        "Hot take: ",
        "Lowkey, ",
    ],
    "spa": [
        "Sinceramente, ",
        "En mi opinión, ",
        "La verdad, ",
        "Siendo honestos, ",
    ],
    "zho": [
        "老实说，",
        "说实话，",
        "在我看来，",
    ],
    "deu": [
        "Ehrlich gesagt, ",
        "Meiner Meinung nach, ",
        "Ganz ehrlich, ",
    ],
    "ara": [
        "بصراحة، ",
        "في رأيي، ",
        "بصراحة يعني، ",
    ],
}

LANG_SUFFIXES = {
    "eng": [" tbh", " fr", " lol", " smh", " imo", " honestly"],
    "spa": [" jaja", " la verdad", " en serio", " de verdad"],
    "zho": [" 哈哈", " 真的", " 啊", " 呢"],
    "deu": [" lol", " ehrlich", " wirklich", " halt"],
    "ara": [" بصراحة", " والله", " صراحة", " جدًّا"],
}


def add_prefix(text: str, lang: str) -> str:
    prefixes = LANG_PREFIXES.get(lang, [])
    if not prefixes:
        return text
    return random.choice(prefixes) + text


def add_suffix(text: str, lang: str) -> str:
    suffixes = LANG_SUFFIXES.get(lang, [])
    if not suffixes:
        return text

    s = random.choice(suffixes)

    # Avoid double-suffixing if it's already there
    if text.strip().endswith(tuple(suffixes)):
        return text

    # Just attach at the end; if it ends with '.', replace that
    if text.endswith("."):
        return text[:-1] + s + "."
    return text + s


def to_lowercase(text: str, lang: str) -> str:
    # Only makes sense for languages with case
    if lang in ["eng", "spa", "deu"]:
        return text.lower()
    return text


def add_extra_punctuation(text: str, lang: str) -> str:
    # If it already ends with some punctuation, leave it
    if text.endswith(
        ("?", "!", ".", "…", "。", "！", "？", "\"", "'", "”", "’")
    ):
        return text

    # Add a bit of "emotional" punctuation
    return text + random.choice([".", "!", "!!", "?!"])


def slight_shuffle(text: str, lang: str) -> str:
    """
    Swap two random words for languages where splitting on spaces is reasonable.
    Skip for Chinese / Arabic to avoid doing dumb stuff.
    """
    if lang not in ["eng", "spa", "deu"]:
        return text

    words = text.split()
    if len(words) < 5:
        return text

    i, j = sorted(random.sample(range(len(words)), 2))
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def get_aug_funcs(lang: str):
    """
    Return a list of augmentation functions appropriate for this language.
    Each function takes (text: str) and returns new_text: str.
    """
    funcs = [
        lambda t: add_prefix(t, lang),
        lambda t: add_suffix(t, lang),
        lambda t: to_lowercase(t, lang),
        lambda t: add_extra_punctuation(t, lang),
    ]
    # Light word-shuffle only for languages where that makes sense
    if lang in ["eng", "spa", "deu"]:
        funcs.append(lambda t: slight_shuffle(t, lang))
    return funcs


def augment_df(df: pd.DataFrame, lang: str, target_size: int) -> pd.DataFrame:
    """
    Augment a DataFrame to a desired size by applying label-preserving
    text transformations.

    Args:
        df: DataFrame with columns ['id', 'text', 'polarization'] (and optionally 'language')
        lang: language code: 'eng', 'spa', 'zho', 'deu', 'ara'
        target_size: desired number of rows after augmentation

    Returns:
        New DataFrame with up to target_size rows, including a 'language' column.
    """
    df = df.copy()

    # Ensure the original data has the language column
    if "language" not in df.columns:
        df["language"] = lang
    else:
        # Optional sanity check: if there is a language column, keep it consistent
        df["language"] = df["language"].fillna(lang)

    orig_n = len(df)
    if orig_n >= target_size:
        return df

    needed_extra = target_size - orig_n
    aug_per_example = ceil(needed_extra / orig_n)
    aug_rows = []
    aug_funcs = get_aug_funcs(lang)

    for _, row in df.iterrows():
        base_id = str(row["id"])
        text = str(row["text"])
        label = row["polarization"]

        for j in range(aug_per_example):
            if len(aug_rows) >= needed_extra:
                break

            func = random.choice(aug_funcs)
            new_text = func(text)
            new_id = f"{base_id}_{lang}_aug{j+1}_{len(aug_rows)}"

            aug_rows.append(
                {
                    "id": new_id,
                    "text": new_text,
                    "polarization": label,
                    "language": lang,
                }
            )

        if len(aug_rows) >= needed_extra:
            break

    aug_df = pd.DataFrame(aug_rows, columns=["id", "text", "polarization", "language"])
    full_df = pd.concat([df, aug_df], ignore_index=True)
    return full_df
