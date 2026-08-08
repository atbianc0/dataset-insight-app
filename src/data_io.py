import csv
from io import BytesIO, StringIO

import pandas as pd

SUPPORTED_FILE_TYPES = ["csv", "tsv", "txt", "xlsx"]
TEXT_FILE_TYPES = {"csv", "tsv", "txt"}
EXCEL_FILE_TYPES = {"xlsx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_COLUMNS = 500
MAX_CELLS = 20_000_000
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


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


def _normalize_and_validate_headers(headers):
    normalized = []
    for index, header in enumerate(headers):
        if pd.isna(header) or not str(header).strip():
            normalized.append(f"Unnamed: {index}")
        else:
            normalized.append(str(header).strip())

    duplicate_names = pd.Index(normalized)[pd.Index(normalized).duplicated()].unique().tolist()
    if duplicate_names:
        raise ValueError(
            "Column names must be unique after trimming whitespace. Duplicate columns: "
            + ", ".join(duplicate_names[:5])
        )
    return normalized


def _validate_table_shape(df):
    rows, columns = df.shape
    if rows > MAX_ROWS:
        raise ValueError(f"The dataset has {rows:,} rows; the limit is {MAX_ROWS:,}.")
    if columns > MAX_COLUMNS:
        raise ValueError(f"The dataset has {columns:,} columns; the limit is {MAX_COLUMNS:,}.")
    if rows * columns > MAX_CELLS:
        raise ValueError(
            f"The dataset has {rows * columns:,} cells; the limit is {MAX_CELLS:,}."
        )


def read_uploaded_table_details(uploaded_file):
    file_name = uploaded_file.name
    suffix = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"The uploaded file is {len(content) / (1024 * 1024):.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

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
        header_row = next(csv.reader(StringIO(decoded), delimiter=delimiter), [])
        _normalize_and_validate_headers(header_row)
        df = pd.read_csv(StringIO(decoded), sep=delimiter)
        source_summary = f"Loaded {suffix.upper()} file with '{delimiter}' as the detected delimiter."
    else:
        excel_file = pd.ExcelFile(BytesIO(content))
        sheet_name = excel_file.sheet_names[0]
        if len(excel_file.sheet_names) > 1:
            warnings.append(
                f"Workbook has {len(excel_file.sheet_names)} sheets. The app used the first sheet: '{sheet_name}'."
            )
        raw_header = pd.read_excel(
            BytesIO(content),
            sheet_name=sheet_name,
            header=None,
            nrows=1,
        )
        if not raw_header.empty:
            _normalize_and_validate_headers(raw_header.iloc[0].tolist())
        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
        source_summary = f"Loaded Excel workbook from sheet '{sheet_name}'."

    df.columns = _normalize_and_validate_headers(df.columns)
    _validate_table_shape(df)

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
    safe = df.copy()

    def escape_formula(value):
        if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
            return "'" + value
        return value

    for column in safe.columns:
        if (
            pd.api.types.is_object_dtype(safe[column])
            or pd.api.types.is_string_dtype(safe[column])
            or isinstance(safe[column].dtype, pd.CategoricalDtype)
        ):
            safe[column] = safe[column].map(escape_formula)

    return safe.to_csv(index=False).encode("utf-8")
