import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import multipletests

def alignment_associations(words_source: list[str], parenth_aligned=None, src_list=None, trg_list=None, topK: int | None = None, min_count: int = 1, missing: str = "NOMATCH", pretokenized: bool = False, fdr_alpha: float | None = None, separate: bool = False):
    """
    Find target-feature associations for selected source words in aligned data.

    Parameters
    ----------
    words_source : list[str]
        Source word(s) to test, e.g. ["time"] or ["time", "when"].

    parenth_aligned : str or iterable[str], optional
        Parenthetical alignment data, e.g. "time (Zeit) when (wenn)".
        Can be a single string, pandas Series, list, etc.

    src_list : iterable[str], optional
        Source-word iterable containing already aligned source words.

    trg_list : iterable[str], optional
        Target-feature iterable corresponding positionally to src_list.

    topK : int, optional
        Maximum number of features to return.

    min_count : int, default=1
        Minimum feature frequency required for inclusion.

    missing : str, default="NOMATCH"
        Target value treated as missing, besides NaN or empty string.

    pretokenized : bool, default=False
        If True, whitespace-separated target features are treated as already
        tokenized, preserving characters such as $ and @.

    fdr_alpha : float, optional
        False Discovery Rate threshold. If None, no FDR correction is applied and q-values are
        returned as NaN. If specified, Benjamini-Hochberg FDR correction is
        applied and only features with q_value <= fdr_alpha are returned.

    separate : bool, default=False
        If True, calculate associations independently for each source word,
        reusing the same vectorized target data. Returns a dictionary of DataFrames.
        If False, treat all selected source words as one group.

    Returns
    -------
    pandas.DataFrame or dict[str, pandas.DataFrame]
        Target-word association statistics including chi2, p_value,
        count, true_pos, false_pos, false_neg, true_neg, precision,
        recall, false_positive_rate, and cramers_V.

    Notes
    -----
    Provide either parenth_aligned or both src_list and trg_list, not both.

    Example
    -------
    alignment_associations(["time"], parenth_aligned=df["targ"], topK=20)
    alignment_associations(["time"], src_list=df["src"], trg_list=df["trg"], topK=20)
    alignment_associations(["time", "when"], parenth_aligned=df["targ"], separate=True)
    """
    rx = re.compile(r"([^\s()]+)\s*\(([^)]*)\)")
    Y, corp = [], []

    if parenth_aligned is not None:
        if isinstance(parenth_aligned, str):
            parenth_aligned = [parenth_aligned]
        for line in parenth_aligned:
            for src, trg in rx.findall(line):
                if pd.isna(trg) or str(trg).strip() in {"", missing}:
                    continue
                corp.append(trg)
                Y.append(src if separate else (1 if src in words_source else 0))
    elif src_list is not None and trg_list is not None:
        if len(src_list) != len(trg_list):
            raise ValueError("src_list and trg_list must have the same length.")
        for src, trg in zip(src_list, trg_list):
            if pd.isna(trg) or str(trg).strip() in {"", missing}:
                continue
            corp.append(trg)
            Y.append(src if separate else (1 if src in words_source else 0))
    else:
        raise ValueError("Provide parenth_aligned, or both src_list and trg_list.")

    Y = np.asarray(Y, dtype=str if separate else int)
    if Y.size == 0 or (not separate and (Y.all() or (~Y.astype(bool)).all())):
        raise ValueError("Data must contain both selected and comparison observations.")

    v = TfidfVectorizer(use_idf=False, norm=None, lowercase=True, token_pattern=r"\S+" if pretokenized else r"\b\w+\b")
    X = v.fit_transform(corp)
    feats = np.array(v.get_feature_names_out())

    source_labels = Y.copy()
    results_by_word = {}
    for word in (words_source if separate else [None]):
        Y = (source_labels == word).astype(int) if separate else source_labels
        if separate and (Y.all() or (~Y.astype(bool)).all()):
            results_by_word[word] = pd.DataFrame()
            continue

        X_current = X
        feats_current = feats

        scores, pvals = chi2(X_current, Y)
        q_values = None
        if fdr_alpha is not None:
            q_values = multipletests(pvals, method="fdr_bh")[1]
            keep = q_values <= fdr_alpha
            X_current, feats_current, scores, pvals, q_values = X_current[:, keep], feats_current[keep], scores[keep], pvals[keep], q_values[keep]

        counts = np.asarray(X_current.sum(axis=0)).ravel()
        N = len(Y)
        results = []
        for i, feat in enumerate(feats_current):
            if counts[i] < min_count:
                continue
            col = X_current[:, i].toarray().ravel() > 0
            TP = np.sum((col == 1) & (Y == 1)) # true positives (i.e. trg word aligned to chosen src word)
            FP = np.sum((col == 1) & (Y == 0)) # false positives (i.e. trg word aligned to something other than chosen src word)
            FN = np.sum((col == 0) & (Y == 1)) # false negatives (i.e. chosen src word aligned to something other than current trg word)
            TN = np.sum((col == 0) & (Y == 0)) # true negatives
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
            association_difference = recall - false_positive_rate
            # Cramér’s V for 2x2 table
            chi2_val = scores[i]
            phi2 = chi2_val / N
            cramers_v = np.sqrt(phi2)
            results.append({
                "feature": feat,
                "chi2": chi2_val,
                "p_value": pvals[i],
                "q_value": q_values[i] if q_values is not None else np.nan,
                "count": counts[i],
                "true_pos": TP,
                "false_pos": FP,
                "false_neg": FN,
                "true_neg": TN,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "association_difference": association_difference,
                "cramers_V": cramers_v})
        df_out = pd.DataFrame(results)
        if df_out.empty:
            if separate:
                results_by_word[word] = df_out
                continue
            return df_out
        df_out = df_out.sort_values("chi2", ascending=False)
        df_out = df_out if topK is None else df_out.head(topK)

        if separate:
            results_by_word[word] = df_out
        else:
            return df_out

    return results_by_word