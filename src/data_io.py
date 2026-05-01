import csv
from io import BytesIO, StringIO

import pandas as pd


SUPPORTED_FILE_TYPES = ["csv", "tsv", "txt", "xlsx", "xls"]
TEXT_FILE_TYPES = {"csv", "tsv", "txt"}
EXCEL_FILE_TYPES = {"xlsx", "xls"}


def _detect_delimiter(sample_text):
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _decode_text_content(content):
    for encoding in ["utf-8-sig", "utf-8"]:
        try:
            return content.decode(encoding), None
        except UnicodeDecodeError:
            continue

    return (
        content.decode("latin-1"),
        "The file was decoded with a latin-1 fallback, so check special characters if text looks unusual.",
    )


def _build_schema_preview(df):
    rows = []
    row_count = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        rows.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing_pct": round(series.isna().mean() * 100, 1),
                "coverage_pct": round((non_null / row_count) * 100, 1),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def read_uploaded_table_details(uploaded_file):
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

    warnings = []
    delimiter = None
    sheet_name = None

    if suffix in TEXT_FILE_TYPES:
        decoded, decode_warning = _decode_text_content(content)
        if decode_warning:
            warnings.append(decode_warning)
        delimiter = _detect_delimiter(decoded[:5000])
        df = pd.read_csv(StringIO(decoded), sep=delimiter)
        source_summary = f"Loaded {suffix.upper()} file with '{delimiter}' as the detected delimiter."
    else:
        excel_file = pd.ExcelFile(BytesIO(content))
        sheet_name = excel_file.sheet_names[0]
        if len(excel_file.sheet_names) > 1:
            warnings.append(
                f"Workbook has {len(excel_file.sheet_names)} sheets. The app used the first sheet: '{sheet_name}'."
            )
        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
        source_summary = f"Loaded Excel workbook from sheet '{sheet_name}'."

    duplicate_columns = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].unique().tolist()
    if duplicate_columns:
        warnings.append(
            "Duplicate column names were found: "
            + ", ".join([str(column) for column in duplicate_columns[:5]])
            + ". Later duplicate columns may be hidden after sanitization."
        )

    if df.empty:
        warnings.append("The uploaded table has headers but no data rows.")

    return {
        "dataframe": df,
        "file_type": suffix,
        "source_summary": source_summary,
        "delimiter": delimiter,
        "sheet_name": sheet_name,
        "warnings": warnings,
        "schema_preview": _build_schema_preview(df),
    }


def read_uploaded_table(uploaded_file):
    return read_uploaded_table_details(uploaded_file)["dataframe"]


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
