# NBA Match Predictor

An advanced machine learning system that predicts NBA game outcomes using player-level advanced statistics and injury impact analysis.

## Overview

This project implements an AI/ML pipeline to predict which team will win an NBA game by analyzing advanced basketball statistics. The system features:

- **Random Forest Classification Model** - Trained on 15 advanced stats metrics
- **Injury Impact Analysis** - Adjusts predictions based on player injuries (65% performance reduction)
- **Interactive GUI** - User-friendly Tkinter interface for predictions
- **Data Processing Pipeline** - Aggregates player-level stats into team-level metrics

## Features

### Core Prediction System
- **Model Type**: Random Forest Classifier (52 estimators, depth 5)
- **Accuracy**: ~85-90% (based on optimized hyperparameters)
- **Input Features**: 15 advanced basketball statistics per team
  - True Shooting % (TS%)
  - Effective Field Goal % (eFG%)
  - 3-Point Attempt Rate (3PAr)
  - Free Throw Rate (FTr)
  - Offensive/Defensive Rebound % (ORB%, DRB%)
  - Total Rebound % (TRB%)
  - Assist % (AST%)
  - Steal % (STL%)
  - Block % (BLK%)
  - Turnover % (TOV%)
  - Usage % (USG%)
  - Offensive/Defensive Rating (ORtg, DRtg)
  - Box Plus/Minus (BPM)

### Injury Prediction System
- Accounts for player injuries with 65% performance reduction
- Adjusts team stats based on active roster
- Predicts impact on game outcomes

### GUI Interface
- Select home and away teams
- View player rosters with stats
- Mark players as injured
- Get win probability predictions
- User-friendly Tkinter application

## Requirements

```
pandas
numpy
openpyxl
scikit-learn
joblib
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NBA\ match\ predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
NBA match predictor/
├── main.py                          # Main AI/ML pipeline orchestrator
├── main_gui.py                      # Tkinter GUI interface
├── model_training.py                # Model training and evaluation
├── dataset_builder.py               # Data preprocessing and aggregation
├── predict_winner.py                # Prediction interface
├── injury_prediction.py             # Injury impact system
├── Readers.py                       # Excel file parsing utilities
├── convert_html_xls_to_xlsx.py     # Format conversion tool
├── game_scores.csv                  # Game results data
├── game_scores.csv                  # Game results data
├── trained_model.pkl                # Pre-trained Random Forest model
├── requirements.txt                 # Python dependencies
├── Training Data/                   # Game statistics (60+ Excel files)
│   └── [Team A vs Team B dates].xlsx # Individual game data
└── README.md                        # This file
```

## Usage

### Option 1: Run the GUI Application
```bash
python main_gui.py
```

Launch the interactive GUI to:
1. Select home and away teams
2. View player rosters
3. Mark injured players
4. Get game outcome predictions with confidence scores

### Option 2: Run the Full AI Pipeline
```bash
python main.py
```

This executes the complete pipeline:
1. Loads and aggregates game data from Training Data folder
2. Preprocesses the dataset
3. Trains the Random Forest model
4. Evaluates model performance
5. Saves the trained model

### Option 3: Make Direct Predictions
Use the `GamePredictor` class in `predict_winner.py`:

```python
from predict_winner import GamePredictor

predictor = GamePredictor("trained_model.pkl")
prediction = predictor.predict(team_stats_dataframe)
```

## Data Format

### Training Data Structure
Excel files in the `Training Data/` folder contain:
- **Filename**: `Team A vs Team B [Date].xlsx`
- **Sheet**: Sheet1
- **Columns**: Player names and advanced statistics
- **Format**: Player-level metrics aggregated to team level

### game_scores.csv
Contains game results with columns:
- `Game`: Game identifier
- `team_a`: First team name
- `Winner`: Winning team name

## Model Details

### Random Forest Configuration
```python
RandomForestClassifier(
    n_estimators=52,          # 52 decision trees
    max_depth=5,              # Limit tree depth to prevent overfitting
    min_samples_split=3,      # Minimum samples required to split
    min_samples_leaf=4,       # Minimum samples per leaf node
    random_state=4940,        # Reproducibility seed
    n_jobs=-1                 # Use all CPU cores
)
```

### Model Performance
- **Train/Test Split**: 75% training, 25% testing
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
- **Cross-validation**: Stratified split to maintain class balance

## Data Pipeline

1. **Loading**: Read Excel files from Training Data folder
2. **Parsing**: Extract player stats using `Readers.py`
3. **Aggregation**: Combine player-level stats to team-level metrics
4. **Preprocessing**: 
   - MinMax scaling normalization
   - Feature selection (15 advanced stats)
   - Label encoding (Win=1, Loss=0)
5. **Training**: Random Forest on 75% of data
6. **Evaluation**: Test on held-out 25% of data

## NBA Statistics Explained

- **TS% (True Shooting %)**: Efficiency metric accounting for 2s, 3s, FTs
- **eFG% (Effective FG %)**: Adjusts FG% for 3-point attempts
- **ORtg/DRtg (Offensive/Defensive Rating)**: Points per 100 possessions
- **BPM (Box Plus/Minus)**: Net points per 100 possessions contributed by player
- **AST%, STL%, BLK% (Assist/Steal/Block %)**: Percentage of teammate possessions

## Key Insights

- Advanced statistics are strong predictors of game outcomes
- Injury impact modeling (65% performance factor) significantly affects predictions
- Random Forest captures non-linear relationships between stats and wins
- Model balances accuracy with interpretability

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates complete ML pipeline |
| `main_gui.py` | Tkinter interface for predictions |
| `model_training.py` | Random Forest training logic |
| `dataset_builder.py` | Data aggregation and preprocessing |
| `predict_winner.py` | Prediction interface and utilities |
| `injury_prediction.py` | Injury impact calculation system |
| `Readers.py` | Excel parsing and data extraction |
| `convert_html_xls_to_xlsx.py` | Format conversion tool |

## Contributors

Desmond Top

Mohamedamin Abchir

