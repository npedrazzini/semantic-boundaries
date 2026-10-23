import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_ngrams(words, ngram_range=(2, 8), position="any", exclude_words=None):
    """
    Extract position-sensitive character n-grams from a sequence of words.

    Parameters
    ----------
    words : iterable[str]
        Words from which character n-grams are extracted.

    ngram_range : tuple[int, int], default=(2, 8)
        Minimum and maximum character n-gram length.

    position : {"any", "initial", "medial", "final", "whole"}, default="any"
        Position of n-grams to retain. Position is encoded in the resulting
        features using $ for word-initial and @ for word-final boundaries.
        Whole-word features contain both markers.

    exclude_words : iterable[str], optional
        Words to exclude from n-gram extraction. Excluded words are represented
        by empty strings, preserving positional correspondence with the input.

    Returns
    -------
    list[str]
        A list positionally corresponding to `words`. Each element contains
        the extracted n-gram features separated by spaces.

    Notes
    -----
    Boundary notation:
        $abc   = word-initial
        abc    = word-medial
        abc@   = word-final
        $abc@  = whole word

    Empty, NaN, and excluded words produce an empty string.

    Example
    -------
    extract_ngrams(["walking", "talking"], ngram_range=(2, 4), position="final")
    """
    exclude_words = set() if exclude_words is None else set(exclude_words)
    out = []
    for word in words:
        if pd.isna(word) or not str(word).strip():
            out.append("")
            continue
        word = str(word).lower()
        if word in exclude_words:
            out.append("")
            continue
        ngrams = []
        for n in range(ngram_range[0], min(ngram_range[1], len(word)) + 1):
            for i in range(len(word) - n + 1):
                gram = word[i:i+n]
                is_initial = i == 0
                is_final = i + n == len(word)
                if is_initial and is_final:
                    feature = f"${gram}@"
                    feature_position = "whole"
                elif is_initial:
                    feature = f"${gram}"
                    feature_position = "initial"
                elif is_final:
                    feature = f"{gram}@"
                    feature_position = "final"
                else:
                    feature = gram
                    feature_position = "medial"
                if position == "any" or position == feature_position:
                    ngrams.append(feature)
        out.append(" ".join(ngrams))
    return out

def cluster_ngrams(df, feature_col="feature", min_count=0, min_true_pos=0, min_precision=0, min_recall=0, max_fpr=1, min_chi2=0, max_p=1, max_q=1, min_cramers_v=0, positive_only=False, topK=20, sort_by="true_pos", eps=1, min_samples=3, ngram_range=(1, 8)):
    """
    Cluster associated character n-grams using DBSCAN.

    Parameters
    ----------
    df : pandas.DataFrame
        Association results, typically returned by `alignment_associations()`.

    feature_col : str, default="feature"
        Column containing the n-gram features to cluster.

    min_count : int, default=0
        Minimum overall feature count required for inclusion.

    min_true_pos : int, default=0
        Minimum number of occurrences among selected source observations.

    min_precision : float, default=0
        Minimum precision required for inclusion.

    min_recall : float, default=0
        Minimum recall required for inclusion.

    max_fpr : float, default=1
        Maximum false-positive rate allowed.

    min_chi2 : float, default=0
        Minimum chi-square statistic required.

    max_p : float, default=1
        Maximum raw p-value allowed.

    max_q : float, default=1
        Maximum FDR-adjusted p-value (q-value) allowed.

    min_cramers_v : float, default=0
        Minimum Cramér's V required.

    positive_only : bool, default=False
        If True, retain only features more frequent proportionally among the
        selected observations than among comparison observations, i.e.
        `association_difference > 0`.

    topK : int or None, default=20
        Maximum number of features to cluster after filtering and sorting.
        If None, all surviving features are used.

    sort_by : str, default="true_pos"
        DataFrame column used to rank features before applying `topK`.

    eps : float, default=1
        DBSCAN neighborhood radius.

    min_samples : int, default=3
        Minimum number of neighboring observations required by DBSCAN for
        a point to be considered a core point.

    ngram_range : tuple[int, int], default=(1, 8)
        Character n-gram range used to create TF-IDF representations of the
        candidate features for clustering.

    Returns
    -------
    pandas.DataFrame
        Filtered association results with an additional `cluster` column.
        DBSCAN assigns -1 to features classified as noise.

    Notes
    -----
    Candidate features are represented with character-level TF-IDF vectors
    before DBSCAN clustering. Statistical filters are applied before
    clustering. Their defaults impose no filtering, except for `topK=20`.

    Example
    -------
    cluster_ngrams(
        associations,
        min_count=10,
        max_q=0.05,
        positive_only=True,
        topK=20
    )
    """
    filtered = df[
        (df["count"] >= min_count) &
        (df["true_pos"] >= min_true_pos) &
        (df["precision"] >= min_precision) &
        (df["recall"] >= min_recall) &
        (df["false_positive_rate"] <= max_fpr) &
        (df["chi2"] >= min_chi2) &
        (df["p_value"] <= max_p) &
        (df["q_value"].isna() | (df["q_value"] <= max_q)) &
        (df["cramers_V"] >= min_cramers_v)
    ].copy()

    if positive_only:
        filtered = filtered[filtered["association_difference"] > 0]
    if filtered.empty:
        return filtered
    if sort_by not in filtered.columns:
        raise ValueError(f"sort_by must be a column in df: {sort_by}")
    filtered = filtered.sort_values(sort_by, ascending=False)
    if topK is not None:
        filtered = filtered.head(topK)
    features = filtered[feature_col].astype(str).tolist()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
    X = vectorizer.fit_transform(features).toarray()
    filtered["cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return filtered