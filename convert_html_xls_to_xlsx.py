import pathlib
import pandas as pd
from bs4 import BeautifulSoup

# This document was entirely AI-generated as a helper to convert old .xls files to .xlsx.

def is_html_file(path: pathlib.Path, n: int = 2048) -> bool:
    try:
        head = path.open("rb").read(n).lstrip()
    except OSError:
        return False
    head_low = head[:16].lower()
    return head_low.startswith(b"<html") or head_low.startswith(b"<!doctype html")

def parse_html_table_with_bs(path: pathlib.Path):
    text = path.read_bytes()
    soup = BeautifulSoup(text, "lxml")  # fallback to 'html.parser' if lxml not available
    table = soup.find("table")
    if table is None:
        return None
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        # prefer td, fallback to th
        for cell in tr.find_all(["td", "th"]):
            cells.append(cell.get_text(strip=True))
        # allow empty rows
        rows.append(cells)
    if not rows:
        return None
    # If first row looks like header, use it as columns when lengths match next row
    header = rows[0]
    data = rows[1:] if any(header) and len(header) == max(len(r) for r in rows[1:]) else rows
    df = pd.DataFrame(data)
    if data is rows[1:]:
        df.columns = header
    return df

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handle MultiIndex columns or tuple-like columns by joining parts with a space
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(p).strip() for p in col if p is not None and str(p).strip() != ""]
            new_cols.append(" ".join(parts) if parts else "")
        df.columns = new_cols
    else:
        # some parsers produce tuple-like single-level columns
        if any(isinstance(c, (list, tuple)) for c in df.columns):
            new_cols = []
            for col in df.columns:
                if isinstance(col, (list, tuple)):
                    parts = [str(p).strip() for p in col if p is not None and str(p).strip() != ""]
                    new_cols.append(" ".join(parts) if parts else "")
                else:
                    new_cols.append(str(col))
            df.columns = new_cols
    return df

def convert_folder(folder: str):
    p = pathlib.Path(folder)
    for path in sorted(p.glob("*.xls")):
        try:
            if not is_html_file(path):
                print(f"Skipping (binary .xls): {path.name}")
                continue
            # try pandas first
            try:
                dfs = pd.read_html(path)
                if not dfs:
                    raise ValueError("No tables found by pandas")
                df = dfs[0]
            except Exception:
                df = parse_html_table_with_bs(path)
                if df is None:
                    print(f"Failed to parse HTML table in {path.name}")
                    continue

            # Flatten MultiIndex/tuple columns so openpyxl can write with index=False
            df = _flatten_columns(df)

            out = path.with_suffix(".xlsx")
            try:
                df.to_excel(out, index=False, engine="openpyxl")
            except NotImplementedError:
                # fallback: write index if writer still objects
                df.to_excel(out, index=True, engine="openpyxl")
            print(f"Converted {path.name} -> {out.name}")
        except Exception as e:
            print(f"Error converting {path.name}: {e}")

def main():
    
    folder = "Training Data"

    print(f"Processing folder: {folder}")
    convert_folder(folder)

if __name__ == "__main__":
    main()