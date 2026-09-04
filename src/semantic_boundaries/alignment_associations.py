import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import re


def alignment_associations(words_source: list[str], parenth_aligned=None, src_list=None, trg_list=None, topK: int | None = None, min_count: int = 1, missing: str = "NOMATCH"):    
    """
    Find target-word associations for selected source words in aligned data.

    Parameters
    ----------
    words_source : list[str]
        Source word(s) to test, e.g. ["time"] or ["time", "when"].

    parenth_aligned : str or iterable[str], optional
        Parenthetical alignment data, e.g. "time (Zeit) when (wenn)".
        Can be a single string, pandas Series, list, etc.

    src_list : iterable[str], optional
        Source-word iterable[str] containing already aligned source words.

    trg_list : iterable[str], optional
        Target-word iterable[str] corresponding positionally to src_list.

    topK : int, optional
        Maximum number of target words to return.

    min_count : int, default=1
        Minimum target-word frequency required for inclusion.

    missing : str, default="NOMATCH"
        Target value treated as missing, besides NaN or empty string (also included by default).

    Returns
    -------
    pandas.DataFrame
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
                Y.append(1 if src in words_source else 0)

    elif src_list is not None and trg_list is not None:
        for src, trg in zip(src_list, trg_list):
            if pd.isna(trg) or str(trg).strip() in {"", missing}: 
                continue
            corp.append(trg)
            Y.append(1 if src in words_source else 0)
    else:
        raise ValueError('Provide parenth_aligned, or both src_list and trg_list.')
    v = TfidfVectorizer(use_idf=False, norm=None, lowercase=True, token_pattern=r"\b\w+\b")
    X = v.fit_transform(corp)
    feats = np.array(v.get_feature_names_out())
    scores, pvals = chi2(X, Y)
    counts = np.asarray(X.sum(axis=0)).ravel()
    N = len(Y)
    Y = np.array(Y)
    results = []
    for i, feat in enumerate(feats):
        if counts[i] < min_count: 
            continue
        col = X[:, i].toarray().ravel() > 0
        TP = np.sum((col == 1) & (Y == 1)) # true positives (i.e. trg word aligned to chosen src word)
        FP = np.sum((col == 1) & (Y == 0)) # false positives (i.e. trg word aligned to something other than chosen src word)
        FN = np.sum((col == 0) & (Y == 1)) # false negatives (i.e. chosen src word aligned to something other than current trg word)
        TN = np.sum((col == 0) & (Y == 0)) # true negatives
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # Cramér’s V for 2x2 table
        chi2_val = scores[i]
        phi2 = chi2_val / N
        cramers_v = np.sqrt(phi2)
        results.append({
            "feature": feat,
            "chi2": chi2_val,
            "p_value": pvals[i],
            "count": counts[i],
            "true_pos": TP,
            "false_pos": FP,
            "false_neg": FN,
            "true_neg": TN,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": FP / (FP + TN) if (FP + TN) > 0 else 0,
            "cramers_V": cramers_v})
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("chi2", ascending=False)
    return df_out if topK is None else df_out.head(topK)