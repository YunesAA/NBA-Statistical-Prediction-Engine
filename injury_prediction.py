"""
Injury Impact Prediction System
Predicts game outcomes considering player injuries (65% performance reduction).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
# Import dataset builders
from dataset_builder import build_training_dataset, load_all_games

# Advanced stats columns used by the model (Exactly 15 features)
STATS_COLUMNS = [
    'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
    'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM'
]

INJURY_FACTOR = 0.65  # Injured player performs at 65% of normal


class InjuryPredictionSystem:
    """System for predicting game outcomes with injury adjustments."""
    
    def __init__(self, model_path="trained_model.pkl"):
        """Initialize with trained model and game data."""
        try:
            self.model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception:
            print(f"Warning: Could not load {model_path}. Make sure to train the model first.")
            self.model = None
        
        # Load directories
        current_dir = Path(__file__).parent
        training_dir = current_dir / "Training Data"
        
        # --- 1. SETUP SCALER (Fixes Feature Mismatch) ---
        print("Building dataset for scaler...")
        self.df_team_stats = build_training_dataset(str(training_dir))
        
        # Clean the dataset to ONLY contain the 15 model features
        # This prevents "X has 16 features" or "Unseen feature: Result"
        df_for_scaler = pd.DataFrame()
        for col in STATS_COLUMNS:
            if col in self.df_team_stats.columns:
                df_for_scaler[col] = self.df_team_stats[col]
            else:
                df_for_scaler[col] = 0.0 # Fill missing with 0
        
        # Fit scaler on exactly 15 columns
        self.scaler = MinMaxScaler()
        self.scaler.fit(df_for_scaler)
        print("Scaler fitted on exactly 15 features.")
        
        # --- 2. BUILD ROSTERS ---
        print("Loading player rosters...")
        self.game_data_list = load_all_games(str(training_dir))
        self.team_rosters = self._build_team_rosters()
        print(f"Loaded rosters for {len(self.team_rosters)} teams")
    
    def _build_team_rosters(self):
        """Build dictionary of team rosters with player stats from raw game data."""
        rosters = {}
        
        for game_df in self.game_data_list:
            if game_df.empty: continue
            if "Team" not in game_df.columns or "Player" not in game_df.columns: continue

            for _, row in game_df.iterrows():
                team = row['Team']
                # Clean name to prevent mismatch issues
                player_name = str(row['Player']).strip()
                
                # Skip invalid entries
                if pd.isna(team) or pd.isna(player_name) or 'team totals' in player_name.lower():
                    continue
                
                if team not in rosters:
                    rosters[team] = {}
                
                # Extract the 15 stats
                player_stats = {col: row[col] for col in STATS_COLUMNS if col in row.index}
                player_stats['name'] = player_name
                
                # Calculate Importance (Minutes) for sorting in GUI
                mp = row.get('MP', 0)
                if mp is None: mp = 0
                player_stats['MP'] = mp
                
                # Store (using dict keys prevents duplicates)
                rosters[team][player_name] = player_stats

        # Convert back to list format
        final_rosters = {}
        for team, players_dict in rosters.items():
            final_rosters[team] = list(players_dict.values())
            
        return final_rosters
    
    def get_teams(self):
        """Get list of all available teams."""
        return sorted(self.team_rosters.keys())
    
    def get_team_roster(self, team):
        """Get roster for a specific team sorted by importance (Minutes Played)."""
        if team not in self.team_rosters:
            return []
        
        roster = []
        for p in self.team_rosters[team]:
            mp = p.get('MP', 0)
            roster.append({
                'name': p['name'],
                'importance': mp,
                'stats': p
            })
        
        # Sort by importance (highest first)
        roster.sort(key=lambda x: x['importance'], reverse=True)
        return roster
    
    def calculate_team_stats(self, team, injured_players=None):
        """Calculate aggregated team stats, reducing specific players' output."""
        if team not in self.team_rosters:
            raise ValueError(f"Team {team} not found")
        
        # Clean input list
        injured_set = {str(n).strip() for n in (injured_players or [])}
        
        player_list = self.team_rosters[team]
        adjusted_stats_list = []
        
        for p_stats in player_list:
            stats_copy = p_stats.copy()
            name = str(stats_copy.get('name', '')).strip()
            
            # Apply injury penalty
            if name in injured_set:
                # DEBUG PRINT: Verify injury is applied
                print(f" -> APPLYING INJURY TO: {name} (Stats x {INJURY_FACTOR})")
                for col in STATS_COLUMNS:
                    if col in stats_copy and isinstance(stats_copy[col], (int, float)):
                        stats_copy[col] = stats_copy[col] * INJURY_FACTOR
            
            adjusted_stats_list.append(stats_copy)
        
        # Aggregate into team stats (Mean)
        team_agg = {}
        for col in STATS_COLUMNS:
            vals = [p[col] for p in adjusted_stats_list if col in p and isinstance(p[col], (int, float))]
            team_agg[col] = np.mean(vals) if vals else 0.0
            
        return team_agg
    
    def predict_matchup(self, team_a, team_b, injured_a=None, injured_b=None):
        """Predict winner based on aggregated stats."""
        # 1. Calculate stats
        print(f"\nCalculating stats for {team_a}...")
        stats_a = self.calculate_team_stats(team_a, injured_a)
        print(f"Calculating stats for {team_b}...")
        stats_b = self.calculate_team_stats(team_b, injured_b)
        
        # 2. Prepare DataFrames (Explicit columns to match Scaler)
        df_input_a = pd.DataFrame([stats_a], columns=STATS_COLUMNS).fillna(0)
        df_input_b = pd.DataFrame([stats_b], columns=STATS_COLUMNS).fillna(0)
            
        # 3. Scale Data (Transform returns numpy array)
        X_a_scaled = self.scaler.transform(df_input_a)
        X_b_scaled = self.scaler.transform(df_input_b)
        
        # Convert back to DataFrame with names to suppress UserWarning
        X_a_final = pd.DataFrame(X_a_scaled, columns=STATS_COLUMNS)
        X_b_final = pd.DataFrame(X_b_scaled, columns=STATS_COLUMNS)
        
        # 4. Predict
        prob_a = self.model.predict_proba(X_a_final)[0]
        prob_b = self.model.predict_proba(X_b_final)[0]
        
        # Logic: prob_a[1] is probability of Team A winning
        score_a = prob_a[1]
        score_b = prob_b[1]
        
        # Normalize
        total = score_a + score_b
        if total == 0: total = 1
        
        final_prob_a = score_a / total
        final_prob_b = score_b / total
        
        winner = team_a if final_prob_a > final_prob_b else team_b
        
        return {
            'predicted_winner': winner,
            'team_a_win_probability': final_prob_a,
            'team_b_win_probability': final_prob_b,
            'confidence': max(final_prob_a, final_prob_b),
            'stats_a': stats_a,
            'stats_b': stats_b,
            'injured_a': injured_a, # Passed back for printing
            'injured_b': injured_b
        }
    
    def print_prediction(self, result):
        """Helper to print results to console (Used by GUI)."""
        print("-" * 30)
        print(f"Prediction: {result['predicted_winner']} wins!")
        print(f"Confidence: {result['confidence']:.2%}")
        print("-" * 30)

    def get_feature_importance(self):
        """Return feature importance for visualization."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(STATS_COLUMNS, self.model.feature_importances_))
        return {}

    # def predict_matchup(self, team_a, team_b, injured_a=None, injured_b=None):
    #     """
    #     Predict winner between two teams with optional injuries.
        
    #     Args:
    #         team_a: First team name
    #         team_b: Second team name
    #         injured_a: List of injured players on team A
    #         injured_b: List of injured players on team B
        
    #     Returns:
    #         Dictionary with prediction results
    #     """
    #     injured_a = injured_a or []
    #     injured_b = injured_b or []
        
    #     # Get adjusted team stats
    #     stats_a = self.calculate_team_stats(team_a, injured_a)
    #     stats_b = self.calculate_team_stats(team_b, injured_b)
        
    #     # Create feature vectors in correct order (must match training order)
    #     features_a = np.array([stats_a.get(stat, 0) for stat in STATS_COLUMNS])
    #     features_b = np.array([stats_b.get(stat, 0) for stat in STATS_COLUMNS])
        
    #     # Get win probabilities for each team
    #     # Model trained on: does this team win (1) or lose (0)?
    #     prob_a = self.model.predict_proba([features_a])[0]
    #     prob_b = self.model.predict_proba([features_b])[0]
        
    #     # prob_a[1] = probability team A wins
    #     # prob_b[1] = probability team B wins
        
    #     # Normalize probabilities to sum to 1 for head-to-head comparison
    #     total_prob = prob_a[1] + prob_b[1]
    #     if total_prob > 0:
    #         team_a_win_prob = prob_a[1] / total_prob
    #         team_b_win_prob = prob_b[1] / total_prob
    #     else:
    #         team_a_win_prob = 0.5
    #         team_b_win_prob = 0.5
        
    #     # Determine winner based on normalized probabilities
    #     winner = team_a if team_a_win_prob > team_b_win_prob else team_b
    #     win_prob = max(team_a_win_prob, team_b_win_prob)
        
    #     # Get feature importance
    #     feature_importance = pd.DataFrame({
    #         'Feature': STATS_COLUMNS,
    #         'Importance': self.model.feature_importances_
    #     }).sort_values('Importance', ascending=False)
        
    #     return {
    #         'team_a': team_a,
    #         'team_b': team_b,
    #         'predicted_winner': winner,
    #         'team_a_win_probability': team_a_win_prob,
    #         'team_b_win_probability': team_b_win_prob,
    #         'confidence': win_prob,
    #         'team_a_prediction': 'WIN' if winner == team_a else 'LOSS',
    #         'team_b_prediction': 'WIN' if winner == team_b else 'LOSS',
    #         'injured_players_team_a': injured_a,
    #         'injured_players_team_b': injured_b,
    #         'team_a_stats': stats_a,
    #         'team_b_stats': stats_b,
    #         'feature_importance': feature_importance
    #     }

def demo():
    """Demo all three use cases."""
    
    print("\n" + "="*80)
    print("INJURY IMPACT PREDICTION SYSTEM - DEMO")
    print("="*80)
    
    # Initialize system
    system = InjuryPredictionSystem("trained_model.pkl")
    teams = system.get_teams()
    
    print(f"\nAvailable teams: {', '.join(teams[:5])}... and {len(teams)-5} more")
    
    # Use Case 1: Normal conditions
    print("\n" + "="*80)
    print("USE CASE 1: NORMAL CONDITIONS (NO INJURIES)")
    print("="*80)
    
    team_a = "Lakers"
    team_b = "Warriors"
    
    result1 = system.predict_matchup(team_a, team_b)
    system.print_prediction(result1)
    
    # Use Case 2: One team with injured player
    print("\n" + "="*80)
    print("USE CASE 2: TEAM WITH ONE INJURED PLAYER")
    print("="*80)
    
    # Get roster to see available players
    roster_a = system.get_team_roster(team_a)
    injured_player = roster_a[0]['name'] if roster_a else "Player_0"
    
    result2 = system.predict_matchup(team_a, team_b, injured_a=[injured_player])
    system.print_prediction(result2)
    
    # Compare impact
    print(f"\n" + "="*80)
    print(f"INJURY IMPACT ANALYSIS: {injured_player} on {team_a}")
    print(f"="*80)
    print(f"Without injury: {result1['team_a']} win probability = {result1['team_a_win_probability']:.2%}")
    print(f"With injury:    {result2['team_a']} win probability = {result2['team_a_win_probability']:.2%}")
    print(f"Difference:     {(result1['team_a_win_probability'] - result2['team_a_win_probability']):.2%}")
    
    # Use Case 3: Both teams with injuries
    print("\n" + "="*80)
    print("USE CASE 3: BOTH TEAMS WITH INJURED PLAYERS")
    print("="*80)
    
    roster_b = system.get_team_roster(team_b)
    injured_player_b = roster_b[0]['name'] if roster_b else "Player_0"
    
    result3 = system.predict_matchup(team_a, team_b, 
                                     injured_a=[injured_player], 
                                     injured_b=[injured_player_b])
    system.print_prediction(result3)


if __name__ == "__main__":
    demo()
