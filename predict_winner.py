"""
Prediction module for the AI model.
Provides a simple interface for Tkinter UI to make predictions.
"""

import joblib
import pandas as pd
import os


class GamePredictor:
    """
    Loads and manages the trained model for making predictions.
    Designed for easy integration with UI.
    """
    
    def __init__(self, model_path: str = "trained_model.pkl"):
        """
        Initialize the predictor with the trained model.
        
        Args:
            model_path: Path to the saved trained_model.pkl file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path
        print(f"Model loaded from {model_path}")
    
    def predict_winner(self, team_stats_df: pd.DataFrame) -> dict:
        """
        Predict the winner for one or more teams.
        
        Args:
            team_stats_df: DataFrame with team statistics
                          Must have columns: Team and advanced stats (TS%, eFG%, ORtg, DRtg, BPM, etc.)
                          Can contain one or two rows (one per team)
        
        Returns:
            Dictionary with predictions and probabilities:
            {
                'predictions': array of 0 or 1 (0=Loss, 1=Win),
                'team_names': list of team names,
                'prediction_text': readable prediction string
            }
        """
        if team_stats_df.empty:
            raise ValueError("DataFrame is empty")
        
        # Extract team names
        team_names = team_stats_df["Team"].tolist() if "Team" in team_stats_df.columns else []
        
        # Drop Team and GameID columns for prediction
        X = team_stats_df.drop(columns=["Team", "GameID"], errors="ignore")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get prediction probabilities if available
        try:
            probabilities = self.model.predict_proba(X)
        except AttributeError:
            probabilities = None
        
        # Format output
        result = {
            'predictions': predictions,
            'team_names': team_names,
            'probabilities': probabilities,
            'prediction_text': self._format_prediction(team_names, predictions, probabilities)
        }
        
        return result
    
    def _format_prediction(self, team_names: list, predictions: list, probabilities=None) -> str:
        """
        Format prediction results into readable text.
        
        Args:
            team_names: List of team names
            predictions: Array of predictions (0 or 1)
            probabilities: Array of prediction probabilities
        
        Returns:
            Formatted string describing the predictions
        """
        if len(predictions) == 1:
            pred_text = "WIN" if predictions[0] == 1 else "LOSS"
            team = team_names[0] if team_names else "Team"
            
            if probabilities is not None:
                conf = max(probabilities[0]) * 100
                return f"{team}: {pred_text} (Confidence: {conf:.1f}%)"
            else:
                return f"{team}: {pred_text}"
        
        elif len(predictions) == 2:
            team1 = team_names[0] if team_names else "Team 1"
            team2 = team_names[1] if len(team_names) > 1 else "Team 2"
            
            if predictions[0] == 1:
                winner = team1
            else:
                winner = team2
            
            if probabilities is not None:
                conf1 = max(probabilities[0]) * 100
                conf2 = max(probabilities[1]) * 100
                return f"Predicted Winner: {winner}\n{team1} Win Prob: {conf1:.1f}%\n{team2} Win Prob: {conf2:.1f}%"
            else:
                return f"Predicted Winner: {winner}"
        
        else:
            return "Invalid number of teams for prediction"
    
    def predict_winner_simple(self, team_stats_df: pd.DataFrame) -> str:
        """
        Simplified prediction function - returns just the prediction text.
        Recommended for UI integration.
        
        Args:
            team_stats_df: DataFrame with team statistics
        
        Returns:
            String with prediction result
        """
        result = self.predict_winner(team_stats_df)
        return result['prediction_text']


# Standalone convenience function
def predict_winner(team_stats_df: pd.DataFrame, model_path: str = "trained_model.pkl") -> dict:
    """
    Convenience function to make a prediction without creating a GamePredictor object.
    
    Usage:
        result = predict_winner(team_stats_df)
        print(result['prediction_text'])
    
    Args:
        team_stats_df: DataFrame with team statistics
        model_path: Path to trained model file
    
    Returns:
        Dictionary with predictions and probabilities
    """
    predictor = GamePredictor(model_path)
    return predictor.predict_winner(team_stats_df)
