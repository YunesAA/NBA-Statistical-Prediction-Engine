from typing import Optional, Dict
import re
from pathlib import Path

import pandas as pd
import numpy as np

STATS_PREFIX = "Advanced Box Score Stats"

def _parse_minutes(value) -> Optional[int]:
    """Convert a 'MM:SS' string into total seconds."""

    if value == "Did Not Play" or value == "Did Not Dress":
        return 0

    try:
        minutes, seconds = value.split(":")

    except Exception:
        #print(f"Warning: Unable to parse minutes value: {value}")
        return value

    return int(minutes) * 60 + int(seconds)


def _parse_percent_like(value) -> Optional[float]:
    """Normalize percent-like values to decimal.
    ".594" -> 0.594, "59.4%" -> 0.594, 59.4 -> 0.594
    Handles "Did Not Play" and "Did Not Dress" as 0.
    """

    if value == "Did Not Play" or value == "Did Not Dress":
        return 0

    # If value is int or float already
    if isinstance(value, (int, float, np.floating, np.integer)):

        v = float(value)
        if v > 1.0:
            return v / 100.0
        else:
            return v
    
    # If value is string
    s = str(value).strip()

    if s.endswith("%"):
        return float(s.rstrip("%")) / 100.0
    
    # try to parse as float
    v = float(s)

    if v > 1.0:
        return v / 100.0
    else:
        return v

def process_game(filepath: str, sheet_name: str) -> pd.DataFrame:
    """
    Read an Excel file and extract columns that begin with "Advanced Box Score Stats".
    Returns a DataFrame with cleaned stat columns (MP as minutes float, percent fields normalized).

    Example columns preserved: MP, TS%, eFG%, 3PAr, FTr, ORB%, DRB%, TRB%, AST%, STL%, BLK%, TOV%, USG%, ORtg, DRtg, BPM
    """

    dataframe = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl", header=0)

    columns_list = list(dataframe.columns)

    player_column = columns_list[0]

    # collect stat columns that start with the prefix "Advanced Box Score Stats"
    stat_columns: Dict[str, str] = {}

    for column in columns_list:
        column_lower = str(column).strip().lower()

        if column_lower.startswith(STATS_PREFIX.lower()):
            # remainder after prefix
            remainder = str(column).strip()[len(STATS_PREFIX):].strip()

            # Normalize stat name (remove leading punctuation like ":" or "-")
            remainder = re.sub(r"^[\:\-\s]+", "", remainder)
            if remainder:
                stat_columns[remainder] = column

    # desired stat order (if present)
    desired_stats = [
        "MP",       # minutes played
        "TS%",      # true shooting percentage
        "eFG%",     # effective field goal percentage
        "3PAr",     # three-point attempt rate
        "FTr",      # free throw rate
        "ORB%",     # offensive rebound percentage
        "DRB%",     # defensive rebound percentage
        "TRB%",     # total rebound percentage
        "AST%",     # assist percentage
        "STL%",     # steal percentage
        "BLK%",     # block percentage
        "TOV%",     # turnover percentage
        "USG%",     # usage percentage
        "ORtg",     # offensive rating
        "DRtg",     # defensive rating
        "BPM",      # box plus/minus
    ]

    # Build output DataFrame
    processed_data = pd.DataFrame()

    # Player column
    processed_data["Player"] = dataframe[player_column].astype(str)

    # Insert Team as the first column
    game_title = Path(filepath).stem
    team_name = game_title.split(" vs ")[0].strip()

    processed_data["Team"] = team_name

    # MP parsed
    if "MP" in stat_columns:
        
        column = dataframe[stat_columns.get("MP")]
        processed_data["MP"] = column.apply(_parse_minutes)
    else:
        processed_data["MP"] = None

    # For each desired stat, read value if column exists and normalize percent-like fields
    # the percent-like stats are weirdly formatted sometimes as decimals or percentages
    percent_like_set = {"TS%", "eFG%", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%"}
    for stat_name in desired_stats:

        if stat_name == "MP":
            continue

        # get corresponding column name
        column_name = stat_columns.get(stat_name)

        # process if column exists
        if column_name is not None:

            # get column
            column = dataframe[column_name]

            # normalize percent-like fields
            if stat_name in percent_like_set:
                processed_data[stat_name] = column.apply(_parse_percent_like)
            else:
                # numeric stats
                # coerce errors to NaN
                processed_data[stat_name] = pd.to_numeric(column, errors="coerce")

        # column does not exist
        else:
            processed_data[stat_name] = None


    # Drop non-player rows
    valid_player_mask = ~processed_data["Player"].str.strip().str.lower().isin({"team totals", "team"})

    # also drop rows where MP indicates DNP or Did Not Dress
    if "MP" in processed_data.columns:
        did_not_play_mask = processed_data["MP"].astype(str).str.lower().str.contains("did not play|dnp|did not dress", na=False)
        valid_player_mask = valid_player_mask & ~did_not_play_mask

    processed_data = processed_data[valid_player_mask].reset_index(drop=True)

    # Reorder columns so Team is the first column
    cols = processed_data.columns.tolist()
    if "Team" in cols:
        cols.insert(0, cols.pop(cols.index("Team")))
        processed_data = processed_data[cols]

    return processed_data


#for now since roster doesnt show up in gui
def collect_team_roster(team_name: str, folder: str = "Training Data") -> list:
    """
    Scans all Excel files in the folder that start with:
        '{team_name} vs ...'
    Extracts all unique players from the Player column.

    Returns a sorted list of unique player names.
    """
    folder_path = Path(folder)

    all_players = set()

    # pattern e.g. "76ers vs *.xlsx"
    for file in folder_path.glob(f"{team_name} vs *.xlsx"):
        try:
            df = process_game(str(file), sheet_name="Sheet1")

            # collect players
            if "Player" in df.columns:
                for name in df["Player"].dropna():
                    clean = str(name).strip()
                    if clean and clean.lower() not in {"team", "team totals"}:
                        all_players.add(clean)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Return sorted list
    return sorted(all_players)

def main():
    # Example usage

    sheet_name = "Sheet1"       # This is the name of the tab, within the Excel file
    filepath = Path.cwd() / "Training Data" / "76ers vs Celtics Oct 22.xlsx"    # Path to the Excel file
    df = process_game(filepath, sheet_name)
    
    # Display full DataFrame without truncation
    # (by default, it omits some columns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.width', 2000)
    print(df)

if __name__ == "__main__":
    main()