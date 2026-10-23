# semantic-boundaries

Utilities for semantic-space analysis, including the construction of boundaries around 2D semantic spaces and the analysis of lexical associations in word-aligned parallel data.

The package currently provides:

- `boundary()`: constructs boundaries around points in a 2D semantic space.
- `alignment_associations()` identifies target-language features associated with one or more selected source words or categories in aligned data. By default, target features are words, but the function can also analyze pre-extracted features such as character n-grams.

## Installation

```bash
pip install semantic-boundaries
```

## Boundary

`boundary()` constructs a boundary around a set of points in a 2D semantic space using kernel density estimation. It identifies low-density regions on a grid around the observed points and adds boundary points that can be used to delimit the occupied semantic space.

```python
import numpy as np
from semantic_boundaries import boundary

P = np.column_stack([x, y])
x_with_boundary, x1, y1, xgrid, ygrid, h0 = boundary(P)
```

`P` must be an `(n, 2)` array containing x/y coordinates.

The boundary can be adjusted using `grid`, `density`, `box_offset`, and `tightness`:

```python
boundary(P, grid=50, density=0.40, box_offset=0.1, tightness="auto")
```

With `tightness="auto"`, the kernel bandwidth is estimated automatically from the distribution of the x-coordinates.

> NB: `boundary()` is a Python implementation of the logic used by the R `boundary()` function in [`qlcVisualize`](https://cran.r-project.org/package=qlcVisualize). Besides different default parameters and minor differences arising from external library support, it follows the same core procedure, i.e., estimating a two-dimensional kernel density surface, identifying grid points below a density threshold, and adding points around the outer extent of the data to delimit empty space.

Returns:

- `x_with_boundary`
- `x1`
- `y1`
- `xgrid`
- `ygrid`
- `h0`


## Alignment associations

alignment_associations() identifies target-language words or other features associated with one or more selected source words in word-aligned data.


```python
from semantic_boundaries import alignment_associations
```

The function accepts either parenthetical alignments:

```python
df = alignment_associations(
    ["time"],
    parenth_aligned=alignments,
    topK=20,
    min_count=10)
```

where alignments have the form:

```text
time (Zeit) when (wenn) when (als)
```

or separate iterables of already aligned source and target words:

```python
df = alignment_associations(
    ["time"],
    src_list=source_words,
    trg_list=target_words,
    topK=20,
    min_count=10)
```

`src_list` and `trg_list` must correspond positionally, i.e., the source and target items at each position are treated as an aligned pair.

By default, target strings are tokenized as words. If `trg_list` already contains whitespace-separated features, use `pretokenized=True` to preserve those features exactly:

```python
df = alignment_associations(
    ["when"],
    src_list=source_words,
    trg_list=preextracted_features,
    pretokenized=True)
```

This is particularly useful together with `extract_ngrams()`, which produces position-sensitive character n-gram features.

### Parameters
| Parameter | Type | Default | Description |
| ---- | ----- | ------ | --- |
| `words_source` | `list[str]` | required | Source word(s) whose target associations should be tested, e.g. `["time"]` or `["time", "when"]`. |
| `parenth_aligned` | `str` or iterable of `str` | `None` | Parenthetical alignment data such as `time (Zeit) when (wenn)`. Use either this argument or `src_list` + `trg_list`. |
| `src_list` | iterable of `str` | `None` | Already aligned source items. Must correspond positionally to `trg_list`. |
| `trg_list` | iterable of `str` | `None` | Already aligned target words or features. Must correspond positionally to `src_list`. |
| `topK` | `int` or `None` | `None` | Maximum number of features to return after ranking by chi-square. `None` returns all features. |
| `min_count` | `int` | `1` | Minimum frequency required for a target feature to be returned. |
| `missing` | `str` | `"NOMATCH"` | Value treated as a missing target alignment. NaN and empty strings are also treated as missing. |
| `pretokenized` | `bool` | `False` | If `True`, whitespace-separated items in `trg_list` are treated as already extracted features rather than being tokenized as ordinary words. This preserves features such as `ca@` and `$ca`. |
| `fdr_alpha` | `float` or `None` | `None` | Optional FDR threshold. If specified, only features with `q_value <= fdr_alpha` are returned. `None` calculates q-values without filtering on them. |
| `separate` | `bool` | `False` | If `True`, calculate associations independently for each source word, returning a dictionary of DataFrames. If `False`, treat all selected source words as one group. |

### Output

The function returns a pandas DataFrame with one row per target feature. When separate=True, the function instead returns a dictionary mapping each source word to its own association DataFrame.

| Column | Meaning |
| ----------- | ---------------|
| `feature` | Target-language word or other target feature being tested (e.g. a character n-gram) |
| `chi2` | Chi-square statistic measuring how strongly the feature's distribution differs between alignments with the selected source word(s) and alignments with other source words. Larger values indicate stronger evidence of an association, but do not indicate its direction. |
| `p_value` | Raw p-value associated with the chi-square statistic: the probability of observing an association at least this strong under the null hypothesis of no association. |
| `q_value` | Benjamini-Hochberg FDR-adjusted p-value, accounting for the fact that many target features are tested simultaneously. |
| `count` | Total number of occurrences of the target feature in the supplied alignment data. |
| `true_pos` | Number of occurrences of the target feature aligned with the selected source word(s). For example, if German `Zeit` is aligned with English `time` 84 times, `true_pos = 84` when testing `["time"]`. |
| `false_pos` | Number of occurrences of the target feature aligned with source words other than the selected word(s). For example, how many times `Zeit` occurs aligned with something other than `time`. |
| `false_neg` | Number of occurrences of the selected source word(s) aligned with target features other than the current one. For example, how many occurrences of `time` are translated by something other than `Zeit`. |
| `true_neg` | Number of other source-word occurrences that are also aligned with something other than the current target feature. |
| `precision` | Of all occurrences of the target feature, the proportion aligned with the selected source word(s). For example: *when German `Zeit` occurs, how often is it aligned with English `time`?* |
| `recall` | Of all occurrences of the selected source word(s), the proportion aligned with the target feature. For example: *when English `time` occurs, how often is it aligned with German `Zeit`?* |
| `false_positive_rate` | Of all occurrences of source words other than the selected word(s), the proportion aligned with the target feature. When the supplied data contain exactly two source categories, the `false_positive_rate` obtained by selecting one is equivalent to the recall obtained by selecting the other. |
| `association_difference` | `recall - false_positive_rate`.  Positive values mean the target feature is proportionally more frequent with the selected source word(s); negative values mean it is proportionally more frequent elsewhere. This indicates the direction of the association, which the chi-square statistic itself does not encode. |
| `cramers_V` | Effect-size measure of the strength of the association, normalized for sample size. Unlike the chi-square statistic, it is intended to describe the magnitude of the relationship rather than simply the statistical evidence for one. |

Because `alignment_associations()` performs a separate chi-square test for every target feature, the returned `q_value` provides a Benjamini-Hochberg False Discovery Rate (FDR)-adjusted p-value.This accounts for the fact that many features are tested simultaneously and is particularly useful when testing large numbers of character n-grams.

By default (`fdr_alpha=None`), q-values are calculated and returned but are not used to filter the results. To return only features that remain significant after FDR correction, specify an FDR threshold:

```python
df = alignment_associations(
    ["when"],
    src_list=source_words,
    trg_list=target_words,
    fdr_alpha=0.05
)
```

With `fdr_alpha=0.05`, only features with `q_value <= 0.05` are returned. Raw and FDR-adjusted significance can also be filtered manually:

```python
df[df["p_value"] <= 0.05]
df[df["q_value"] <= 0.05]
```

The output when `separate=True` can be accessed like this:

```python
dfs = alignment_associations(
    ["time", "when"],
    parenth_aligned=alignments,
    separate=True)

time_assoc = dfs["time"]
when_assoc = dfs["when"]
```

### Comparison requirements

`alignment_associations()` compares the selected source word(s) with all unselected source words in the supplied data. There must therefore be a comparison group for the association statistics to make sense.


For example, suppose the original corpus has been filtered to retain only word-level alignments involving the source words `time` and `when`:

```text
src     trg
time    Zeit
when    wenn
when    als
time    Mal
...
```

In this dataset, every source observation is either `time` or `when`. To compare their target-language distributions, select only one of them:

```python
alignment_associations(["time"], src_list=src, trg_list=trg)
```

or:

```python
alignment_associations(["when"], src_list=src, trg_list=trg)
```

Here, selecting `["time"]` makes `time` the selected category and `when` the comparison category (and vice versa).

By contrast, selecting both words can be meaningful when the supplied alignment data come from a broader parallel corpus containing alignments for many other source words:

```text
src     trg
time    Zeit
when    wenn
house   Haus
go      gehen
day     Tag
...
```

or

```text
the (das) book (buch) of (NOMATCH) the (das) genealogy (geschichte) of (NOMATCH) jesus (jesu) 
abraham (abraham) was (NOMATCH) the (NOMATCH) father (zeugte) 
and (und) being (da) warned (ihnen) in (im) a (NOMATCH) dream (traum)
...
```

In either case, selecting both words:

```python
alignment_associations(["time", "when"], src_list=src, trg_list=trg)
```

or, for parenthetical alignments:

```python
alignment_associations(["time", "when"], parenth_aligned=alignments)
```

compares target words aligned with `time` or `when` against target words aligned with all other source words in the supplied data.

### Examples of filtering and ranking

Return the target words with the strongest chi-square association:

```python
df.sort_values("chi2", ascending=False).head(20)
```

Rank by effect size:

```python
df.sort_values("cramers_V", ascending=False).head(20)
```

Find target words with relatively high coverage of the selected source category and little use with the comparison category:

```python
df[(df["recall"] >= 0.20) & (df["false_positive_rate"] <= 0.05)].sort_values("recall", ascending=False)
```

Return only the target words satisfying those criteria:

```python
words = list(df.loc[(df["recall"] >= 0.20) & (df["false_positive_rate"] <= 0.05),"feature"])
```

Apply minimum aligned-frequency, precision, and significance criteria (thresholds should be chosen according to the size, composition, and purpose of the dataset):

```python
words = list(df.loc[(df["true_pos"] >= 10) & (df["precision"] >= 0.30) & (df["p_value"] < 0.05),"feature"])
```

Retain associations that remain significant after FDR correction:

```python
df[df["q_value"] <= 0.05]
```

Retain only features positively associated with the selected source word(s):

```python
df[df["association_difference"] > 0]
```

Rank positive associations by the difference between their rate with the selected source word(s) and their rate with other source words:

```python 
df[df["association_difference"] > 0].sort_values(
    "association_difference",
    ascending=False
).head(20)
```

## N-gram analysis

- `extract_ngrams()` extracts character n-grams from target-language words. 
- `cluster_ngrams()` groups associated n-grams according to their character-level similarity using DBSCAN.

```python
from semantic_boundaries import extract_ngrams, cluster_ngrams
```

### `extract_ngrams()`

For example:

```python
ngrams = extract_ngrams(
    target_words,
    ngram_range=(2, 8),
    position="any"
)
```

To allow for only looking for suffixes and prefixes, the position of each n-gram is encoded using `$` for the beginning of a word and `@` for the end:

| Feature | Meaning                                 |
| ------- | --------------------------------------- |
| `$abc`  | `abc` occurs at the beginning of a word |
| `abc`   | `abc` occurs inside a word              |
| `abc@`  | `abc` occurs at the end of a word       |
| `$abc@` | `abc` constitutes the whole word        |

For example:

```python
ngrams = extract_ngrams(
    target_words,
    ngram_range=(2, 8),
    position="final"
)
```

extracts only word-final n-grams.

Known full-word counterparts can be excluded from n-gram extraction:

```python
ngrams = extract_ngrams(
    target_words,
    position="final",
    exclude_words=["quepaucua", "mericüsü"]
)
```

Excluded, empty, and missing words produce empty strings so that the output remains positionally corresponding to the input.

### Parameters

| Parameter | Type | Default | Description |
| -------- | --------- | -------- | ---------- |
| `words` | iterable of `str` | required | Target-language words from which character n-grams are extracted. |
| `ngram_range` | tuple of `int` | `(2, 8)` | Minimum and maximum length of the character n-grams to extract. |
| `position` | `str` | `"any"` | Position of n-grams to retain. Options are `"any"`, `"initial"`, `"medial"`, `"final"`, and `"whole"`. |
| `exclude_words` | iterable of `str` or `None` | `None` | Words that should not be decomposed into character n-grams.                                       |

The extracted n-grams can be supplied to `alignment_associations()` as pre-extracted features:

```python
associations = alignment_associations(
    ["when"],
    src_list=source_words,
    trg_list=ngrams,
    pretokenized=True
)
```

Here, `pretokenized=True` preserves features such as `ca@` and `$ca` rather than treating them as ordinary words.

### `cluster_ngrams()`

`cluster_ngrams()` takes an association-results DataFrame, filters the associated features according to the supplied criteria, and groups the surviving features using character-level TF-IDF and DBSCAN.

```python
clustered = cluster_ngrams(associations)
```

### Parameters

| Parameter | Type | Default | Description |
| ----------- | --------- | -------- | ------- |
| `df` | `pandas.DataFrame` | required | Association results, typically returned by `alignment_associations()`. |
| `feature_col` | `str` | `"feature"` | Column containing the features to cluster. |
| `min_count` | `int` | `0` | Minimum total frequency required for a feature to enter clustering. |
| `min_true_pos` | `int` | `0` | Minimum number of occurrences aligned with the selected source word(s). |
| `min_precision` | `float` | `0` | Minimum precision required. |
| `min_recall` | `float` | `0` | Minimum recall required. |
| `max_fpr` | `float` | `1` | Maximum false-positive rate allowed. |
| `min_chi2` | `float` | `0` | Minimum chi-square statistic required. |
| `max_p` | `float` | `1` | Maximum raw p-value allowed. |
| `max_q` | `float` | `1` | Maximum FDR-adjusted q-value allowed. |
| `min_cramers_v` | `float` | `0` | Minimum Cramér's V required. |
| `positive_only` | `bool` | `False` | If `True`, retain only features that are proportionally more frequent with the selected source word(s) than with other source words. |
| `topK` | `int` or `None` | `20` | Maximum number of features to pass to DBSCAN after filtering and sorting. `None` uses all surviving features. |
| `sort_by` | `str` | `"true_pos"` | Column used to rank features before applying `topK`. |
| `eps` | `float` | `1` | DBSCAN neighborhood radius. |
| `min_samples` | `int` | `3` | Minimum number of nearby features required for a DBSCAN core point. |
| `ngram_range` | tuple of `int` | `(1, 8)` | Character n-gram range used to represent the candidate features for clustering. This does not determine which linguistic n-grams are extracted; that is controlled by `extract_ngrams()`. |

For example, to retain relatively frequent, positively associated features that survive FDR correction:

```python
clustered = cluster_ngrams(
    associations,
    min_count=10,
    max_q=0.05,
    positive_only=True,
    topK=20
)
```

The statistical filters are independent and can be combined as needed. Setting them to their permissive defaults imposes no filtering.

The function returns the filtered association DataFrame with an additional `cluster` column. A cluster value of `-1` indicates that DBSCAN classified the feature as noise.

Clusters can be inspected with:

```python
for cluster, group in clustered[clustered["cluster"] >= 0].groupby("cluster"):
    print(f"Cluster {cluster + 1}: {group["feature"].tolist()}")
```

## Alignment utils

`prepare_alignments()` converts parenthetical word alignments into two positionally corresponding source and target lists.

```python
from semantic_boundaries import prepare_alignments
```

For example:

```text
time (Zeit) when (wenn) when (NOMATCH)
```

can be converted with:

```python
source_words, target_words = prepare_alignments(alignments)
```

giving:

```python
source_words = ["time", "when", "when"]
target_words = ["Zeit", "wenn", ""]
```

Empty strings and missing values (e.g. NaN) are treated as missing target alignments. By default, the string `NOMATCH` is also treated as missing; a different corpus-specific missing-alignment marker can be specified with missing. Missing target alignments are represented as empty strings in the returned `target_list`.

### Parameters

| Parameter | Type | Default | Description |
| ------- | ------ | ------- | ----- |
| `parenth_aligned` | `str` or iterable of `str` | required | Parenthetical alignment data, e.g. `time (Zeit) when (wenn)`. A single string, pandas Series, list, or other iterable of alignment strings can be supplied. |
| `missing` | `str` | `"NOMATCH"` | Additional value treated as a missing target alignment, besides empty strings and missing values such as `NaN`. Missing target alignments are returned as empty strings. |

### Output

The function returns:

```python
source_words, target_words
```

where `source_words` and `target_words` are positionally corresponding lists: the items at each position represent an aligned source-target pair.

For example:

```python
source_words[0] == "time"
target_words[0] == "Zeit"
```

The function only prepares the alignment data. It does not perform association analysis or other linguistic processing.

