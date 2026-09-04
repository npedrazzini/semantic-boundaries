# semantic-boundaries

Utilities for semantic-space analysis, including the construction of boundaries around 2D semantic spaces and the analysis of lexical associations in word-aligned parallel data.

The package currently provides:

- `boundary()`: constructs boundaries around points in a 2D semantic space.
- `alignment_associations()`: identifies target-word associations in word-aligned parallel data.

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

`alignment_associations()` identifies target-language words associated with one or more selected source words in word-aligned data.

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

### Output

The function returns a pandas DataFrame with one row per target word:

| Column | Meaning |
| --- | --- |
| `feature` | Target-language word |
| `chi2` | Chi-square statistic measuring dependence between occurrence of the target word and membership in the selected vs. unselected source categories |
| `p_value` | p-value associated with the chi-square statistic |
| `count` | Total frequency of the target word |
| `true_pos` | Target-word occurrences aligned with the selected source word(s) |
| `false_pos` | Target-word occurrences aligned with other source words |
| `false_neg` | Selected-source occurrences aligned with other target words |
| `true_neg` | Other-source occurrences aligned with other target words |
| `precision` | Of all occurrences of the target word, the proportion aligned with the selected source word(s) |
| `recall` | Of all occurrences of the selected source word(s), the proportion aligned with the target word |
| `false_positive_rate` | Of all occurrences of other source words, the proportion aligned with the target word. When the supplied data contain exactly two source categories, the `false_positive_rate` obtained by selecting one category is equivalent to the recall obtained by selecting the other. |
| `cramers_V` | Effect size measuring the strength of association between the target word and the selected vs. unselected source categories |


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

Apply minimum aligned-frequency, precision, and significance criteria:

```python
words = list(df.loc[(df["true_pos"] >= 10) & (df["precision"] >= 0.30) & (df["p_value"] < 0.05),"feature"])
```

Thresholds should be chosen according to the size, composition, and purpose of the dataset.