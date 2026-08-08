from io import BytesIO

import pandas as pd
import pytest

from src import data_io
from src.data_io import dataframe_to_csv_bytes, read_uploaded_table, read_uploaded_table_details


class UploadedFileShim:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_read_uploaded_table_details_reads_delimited_text_and_schema_preview():
    content = b"account_id;region;value\n1;north;10\n2;south;15\n"

    details = read_uploaded_table_details(UploadedFileShim("sample.csv", content))

    assert details["dataframe"].columns.tolist() == ["account_id", "region", "value"]
    assert details["delimiter"] == ";"
    assert "detected delimiter" in details["source_summary"].lower()
    assert details["schema_preview"]["column"].tolist() == ["account_id", "region", "value"]


def test_read_uploaded_table_details_reads_first_excel_sheet_and_warns():
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"region": ["north", "south"], "value": [10, 20]}).to_excel(
            writer,
            sheet_name="Primary",
            index=False,
        )
        pd.DataFrame({"unused": [1, 2]}).to_excel(
            writer,
            sheet_name="Secondary",
            index=False,
        )

    details = read_uploaded_table_details(UploadedFileShim("sample.xlsx", buffer.getvalue()))

    assert details["sheet_name"] == "Primary"
    assert details["dataframe"].columns.tolist() == ["region", "value"]
    assert any("first sheet" in warning.lower() for warning in details["warnings"])


def test_duplicate_headers_after_trimming_are_rejected():
    content = b"value, value\n1,2\n"

    with pytest.raises(ValueError, match="unique after trimming"):
        read_uploaded_table_details(UploadedFileShim("sample.csv", content))


def test_legacy_xls_is_not_advertised_as_supported():
    with pytest.raises(ValueError, match="Unsupported file type 'xls'"):
        read_uploaded_table_details(UploadedFileShim("sample.xls", b"not-an-xls"))


def test_table_shape_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(data_io, "MAX_ROWS", 1)

    with pytest.raises(ValueError, match="limit is 1"):
        read_uploaded_table_details(
            UploadedFileShim("sample.csv", b"value\n1\n2\n")
        )


def test_csv_export_neutralizes_spreadsheet_formulas():
    frame = pd.DataFrame(
        {
            "customer": ["=HYPERLINK(\"https://example.test\")", "safe"],
            "score": [1, -2],
        }
    )

    exported = dataframe_to_csv_bytes(frame).decode("utf-8")

    assert "'=HYPERLINK" in exported
    assert "safe" in exported
    assert ",-2" in exported


def test_empty_and_oversized_uploads_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="empty"):
        read_uploaded_table_details(UploadedFileShim("empty.csv", b""))

    monkeypatch.setattr(data_io, "MAX_UPLOAD_BYTES", 3)
    with pytest.raises(ValueError, match="limit"):
        read_uploaded_table_details(UploadedFileShim("large.csv", b"value\n1\n"))


def test_latin1_fallback_blank_header_and_header_only_warning():
    latin = read_uploaded_table_details(
        UploadedFileShim("latin.csv", b",name\n1,caf\xe9\n")
    )
    assert latin["dataframe"].iloc[0, 1] == "café"
    assert str(latin["dataframe"].columns[0]).startswith("Unnamed")
    assert any("latin-1" in warning for warning in latin["warnings"])

    header_only = read_uploaded_table_details(
        UploadedFileShim("headers.tsv", b"first\tsecond\n")
    )
    assert header_only["dataframe"].empty
    assert header_only["delimiter"] == "\t"
    assert any("no data rows" in warning for warning in header_only["warnings"])
    assert read_uploaded_table(UploadedFileShim("one.csv", b"value\n1\n")).iloc[0, 0] == 1


def test_column_and_cell_limits_are_both_enforced(monkeypatch):
    monkeypatch.setattr(data_io, "MAX_COLUMNS", 1)
    with pytest.raises(ValueError, match="columns; the limit is 1"):
        read_uploaded_table_details(UploadedFileShim("wide.csv", b"a,b\n1,2\n"))

    monkeypatch.setattr(data_io, "MAX_COLUMNS", 500)
    monkeypatch.setattr(data_io, "MAX_CELLS", 2)
    with pytest.raises(ValueError, match="cells; the limit is 2"):
        read_uploaded_table_details(UploadedFileShim("cells.csv", b"a,b\n1,2\n3,4\n"))


def test_single_sheet_excel_has_no_multi_sheet_warning():
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"value": [1]}).to_excel(writer, index=False)

    details = read_uploaded_table_details(UploadedFileShim("one.xlsx", buffer.getvalue()))

    assert details["dataframe"].to_dict(orient="list") == {"value": [1]}
    assert not details["warnings"]
