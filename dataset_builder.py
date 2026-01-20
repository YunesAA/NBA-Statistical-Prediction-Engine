import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Readers import process_game

# Load game results from CSV
_game_scores_df = pd.read_csv("game_scores.csv")
_game_results = {}
for _, row in _game_scores_df.iterrows():
    game = row['Game']
    team_a = row['team_a']
    winner = row['Winner']
    # 1 if team_a wins, 0 if team_b wins
    _game_results[game] = 1 if team_a == winner else 0


def extract_teams_from_filename(filename: str) -> tuple:
    """Extract team names from filename 'TeamA vs TeamB.xlsx'"""
    name_without_ext = filename.replace(".xlsx", "").replace(".XLSX", "")
    if " vs " in name_without_ext.lower():
        parts = name_without_ext.split(" vs ")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return None, None


def load_all_games(folder_path: str):
    """Load all Excel files in a folder and add GameID / TeamName"""
    game_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            full_path = os.path.join(folder_path, file)
            try:
                df = process_game(full_path, "Sheet1")
                team_a, team_b = extract_teams_from_filename(file)
                if team_a is None or team_b is None:
                    print(f"WARNING: Could not extract team names from {file}")
                    continue
                df["GameID"] = file
                df["TeamName"] = df["Team"].apply(lambda x: team_a if x == df["Team"].iloc[0] else team_b)
                game_dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return game_dfs


def aggregate_team_stats(df, game_id: str = ""):
    """Aggregate player-level stats into team-level stats"""
    if "TeamName" in df.columns:
        df["Team"] = df["TeamName"]

    if "Team" not in df.columns:
        raise ValueError("DataFrame must include a 'Team' column")

    final_rows = []
    for team in df["Team"].unique():
        team_df = df[df["Team"] == team]
        agg_row = team_df.mean(numeric_only=True)
        agg_row["Team"] = team
        agg_row["GameID"] = game_id
        
        # Get result from loaded game_scores.csv
        result = _game_results.get(game_id, None)
        agg_row["Result"] = result
        
        final_rows.append(agg_row)

    return pd.DataFrame(final_rows)


def build_training_dataset(folder_path: str) -> pd.DataFrame:
    """
    Loads all games from Training Data folder and builds team-level dataset.
    Team names extracted from filenames in format: "TeamA vs TeamB.xlsx"
    """
    game_dfs = load_all_games(folder_path)
    all_team_rows = []

    for game in game_dfs:
        # Extract GameID from the first row (it's the same for all rows in this game)
        game_id_with_ext = game["GameID"].iloc[0] if len(game) > 0 else ""
        # Remove .xlsx extension to match keys in game_scores.csv
        game_id = game_id_with_ext.replace(".xlsx", "").replace(".XLSX", "")
        team_rows = aggregate_team_stats(game, game_id)
        all_team_rows.append(team_rows)

    final_df = pd.concat(all_team_rows, ignore_index=True)
    return final_df



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaNs and remove DNP players"""
    if "minutes" in df.columns:
        df = df[df["minutes"] > 0]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    return df


def normalize_features(df: pd.DataFrame, scaler=None, fit=True) -> tuple:
    """MinMax scale numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    if fit:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_numeric)
    else:
        scaled_data = scaler.transform(df_numeric)

    df_scaled = pd.DataFrame(scaled_data, columns=numeric_cols, index=df.index)

    # Add back non-numeric columns
    for col in df.columns:
        if col not in numeric_cols:
            df_scaled[col] = df[col].values

    return df_scaled, scaler


def preprocess_dataset(df: pd.DataFrame, scaler=None) -> tuple:
    """Full preprocessing pipeline"""
    # Remove rows where Result couldn't be found in game_scores.csv
    df = df[df["Result"].notna()].copy()
    
    # Ensure Result is integer type
    df["Result"] = df["Result"].astype(int)
    
    df = handle_missing_values(df)
    df, scaler = normalize_features(df, scaler=scaler, fit=(scaler is None))
    return df, scaler


def split_dataset(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> tuple:
    """Split into train/test sets (75/25 stratified split)"""
    # Remove rows with missing Result values
    df = df[df["Result"].notna()]
    
    X = df.drop(columns=["Result", "Team", "GameID"], errors="ignore")
    y = df["Result"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
