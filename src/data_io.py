import csv
from io import StringIO

import pandas as pd


SUPPORTED_FILE_TYPES = ["csv", "tsv", "txt"]


def _detect_delimiter(sample_text):
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_uploaded_table(uploaded_file):
    file_name = uploaded_file.name
    suffix = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("The uploaded file is empty.")

    if suffix not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Please upload one of: "
            + ", ".join(SUPPORTED_FILE_TYPES)
        )

    decoded = content.decode("utf-8", errors="ignore")
    delimiter = _detect_delimiter(decoded[:5000])
    return pd.read_csv(StringIO(decoded), sep=delimiter)


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
