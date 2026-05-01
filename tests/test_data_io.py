from io import BytesIO

import pandas as pd

from src.data_io import read_uploaded_table_details


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
