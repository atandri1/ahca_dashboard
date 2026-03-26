from __future__ import annotations

import re
from typing import Callable, Tuple

import numpy as np
import pandas as pd


def clean_text_basic(s: object) -> str:
    s = s if isinstance(s, str) else ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_tfidf_similarity(
    citations: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    citation_text_col: str = "inspection_text",
    tag_col: str = "deficiency_tag",
    baseline_text_col: str = "text",
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> pd.Series:
    """
    Computes row-wise cosine similarity between each citation's text and its matching F-tag regulatory baseline text.

    Returns a float Series aligned to `citations` with NaN where no baseline is available.
    Requires scikit-learn.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception as e:
        raise RuntimeError("scikit-learn is required for TF-IDF similarity. Install `scikit-learn`.") from e

    if citations.empty:
        return pd.Series(dtype="float64")
    if baseline.empty:
        return pd.Series([np.nan] * len(citations), index=citations.index, dtype="float64")
    if tag_col not in citations.columns or citation_text_col not in citations.columns:
        return pd.Series([np.nan] * len(citations), index=citations.index, dtype="float64")

    baseline_df = baseline.copy()
    if "deficiency_tag" not in baseline_df.columns:
        raise ValueError("Baseline must include `deficiency_tag` (int).")

    baseline_df["deficiency_tag"] = pd.to_numeric(baseline_df["deficiency_tag"], errors="coerce").astype("Int64")
    baseline_df = baseline_df.dropna(subset=["deficiency_tag"])
    baseline_df["text_clean"] = baseline_df[baseline_text_col].apply(clean_text_basic)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, min_df=min_df)
    X_base = vectorizer.fit_transform(baseline_df["text_clean"].astype(str))

    tag_to_row = {int(t): i for i, t in enumerate(baseline_df["deficiency_tag"].astype(int).tolist())}
    cit_tags = pd.to_numeric(citations[tag_col], errors="coerce")
    cit_idx = cit_tags.map(tag_to_row)

    texts = citations[citation_text_col].apply(clean_text_basic)
    X_cit = vectorizer.transform(texts.astype(str))

    out = np.full(len(citations), np.nan, dtype=float)
    mask = cit_idx.notna().to_numpy()
    if not mask.any():
        return pd.Series(out, index=citations.index, dtype="float64")

    base_rows = cit_idx[mask].astype(int).to_numpy()
    X_base_sel = X_base[base_rows]
    X_cit_sel = X_cit[mask]

    # Row-wise cosine similarity between corresponding rows.
    numerator = X_base_sel.multiply(X_cit_sel).sum(axis=1).A1
    base_norm = np.sqrt(X_base_sel.multiply(X_base_sel).sum(axis=1)).A1
    cit_norm = np.sqrt(X_cit_sel.multiply(X_cit_sel).sum(axis=1)).A1
    sim = numerator / (base_norm * cit_norm + 1e-12)

    out[mask] = sim
    return pd.Series(out, index=citations.index, dtype="float64")


def compute_sentence_transformer_similarity(
    citations: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    citation_text_col: str = "inspection_text",
    tag_col: str = "deficiency_tag",
    baseline_text_col: str = "text",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    progress_callback: Callable[[float], None] | None = None,
) -> pd.Series:
    """
    Computes cosine similarity using sentence-transformers embeddings.

    This is much slower and requires heavy dependencies. It's intended for small, filtered subsets.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for embedding similarity. Install `sentence-transformers`."
        ) from e

    if citations.empty:
        return pd.Series(dtype="float64")
    if baseline.empty:
        return pd.Series([np.nan] * len(citations), index=citations.index, dtype="float64")
    if tag_col not in citations.columns or citation_text_col not in citations.columns:
        return pd.Series([np.nan] * len(citations), index=citations.index, dtype="float64")

    baseline_df = baseline.copy()
    if "deficiency_tag" not in baseline_df.columns:
        raise ValueError("Baseline must include `deficiency_tag` (int).")

    baseline_df["deficiency_tag"] = pd.to_numeric(baseline_df["deficiency_tag"], errors="coerce").astype("Int64")
    baseline_df = baseline_df.dropna(subset=["deficiency_tag"])
    baseline_df["text_clean"] = baseline_df[baseline_text_col].apply(clean_text_basic)

    if progress_callback is not None:
        progress_callback(0.0)

    model = SentenceTransformer(model_name)

    # Baseline embeddings are usually small, but can still take a few seconds on first model load.
    base_emb = model.encode(
        baseline_df["text_clean"].astype(str).tolist(),
        show_progress_bar=False,
        batch_size=batch_size,
    )
    base_emb = _l2_normalize(np.asarray(base_emb, dtype=float))
    if progress_callback is not None:
        progress_callback(0.2)

    tag_to_row = {int(t): i for i, t in enumerate(baseline_df["deficiency_tag"].astype(int).tolist())}
    cit_tags = pd.to_numeric(citations[tag_col], errors="coerce")
    cit_idx = cit_tags.map(tag_to_row)

    texts = citations[citation_text_col].apply(clean_text_basic).astype(str).tolist()

    # Encode in batches so callers (Streamlit) can show progress.
    if not texts:
        cit_emb = np.zeros((0, base_emb.shape[1]), dtype=float)
    else:
        chunks: list[np.ndarray] = []
        n = len(texts)
        for start in range(0, n, batch_size):
            chunk = texts[start : start + batch_size]
            emb_chunk = model.encode(chunk, show_progress_bar=False, batch_size=batch_size)
            chunks.append(np.asarray(emb_chunk, dtype=float))

            if progress_callback is not None:
                done = min(start + batch_size, n)
                frac = done / n
                progress_callback(0.2 + 0.7 * frac)

        cit_emb = np.vstack(chunks)

    cit_emb = _l2_normalize(np.asarray(cit_emb, dtype=float))
    if progress_callback is not None:
        progress_callback(0.95)

    out = np.full(len(citations), np.nan, dtype=float)
    mask = cit_idx.notna().to_numpy()
    if not mask.any():
        return pd.Series(out, index=citations.index, dtype="float64")

    base_rows = cit_idx[mask].astype(int).to_numpy()
    sim = np.sum(base_emb[base_rows] * cit_emb[mask], axis=1)
    out[mask] = sim
    if progress_callback is not None:
        progress_callback(1.0)
    return pd.Series(out, index=citations.index, dtype="float64")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-12)


def similarity_interpretation(sim: float | None) -> str:
    if sim is None or (isinstance(sim, float) and np.isnan(sim)):
        return "Unavailable"
    if sim >= 0.7:
        return "Very High Alignment"
    if sim >= 0.5:
        return "High Alignment"
    if sim >= 0.3:
        return "Moderate Alignment"
    return "Low Alignment"


def state_similarity_zscores(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    if df.empty or sim_col not in df.columns or "state" not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[sim_col], errors="coerce")
    overall_mean = float(s.mean())
    overall_std = float(s.std())
    stats = (
        pd.DataFrame({"state": df["state"], sim_col: s})
        .dropna(subset=["state"])
        .groupby("state")[sim_col]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    if overall_std and overall_std > 0:
        stats["z_score"] = (stats["mean"] - overall_mean) / overall_std
    else:
        stats["z_score"] = np.nan
    stats = stats.sort_values("z_score")
    return stats
