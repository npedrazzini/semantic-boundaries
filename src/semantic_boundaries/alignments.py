import re
import pandas as pd

def prepare_alignments(parenth_aligned, missing="NOMATCH"):
    """
    Converts parenthetical word alignments into parallel source and target lists.

    Parameters
    ----------
    parenth_aligned : str or iterable[str]
        Parenthetical alignment data, e.g. "time (Zeit) when (wenn)".
        Can be a single string, pandas Series, list, etc.

    missing : str, default="NOMATCH"
        Target value used to represent an unaligned source word. Missing
        targets are converted to empty strings.

    Returns
    -------
    tuple[list[str], list[str]]
        Two positionally corresponding lists containing the source words
        and their aligned target words.

    Example
    -------
    source_list, target_list = prepare_alignments(["time (Zeit) when (wenn)", "when (NOMATCH)"])
    """
    rx = re.compile(r"([^\s()]+)\s*\(([^)]*)\)")
    if isinstance(parenth_aligned, str):
        parenth_aligned = [parenth_aligned]
    source_list, target_list = [], []
    for line in parenth_aligned:
        for src, trg in rx.findall(line):
            source_list.append(src)
            target_list.append("" if pd.isna(trg) or str(trg).strip() in {"", missing} else trg)
    return source_list, target_list