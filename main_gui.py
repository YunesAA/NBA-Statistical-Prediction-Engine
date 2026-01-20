import tkinter as tk
from tkinter import ttk, messagebox
from injury_prediction import InjuryPredictionSystem
from Readers import collect_team_roster

print("Initializing System...")
try:
    # Initialize the backend system
    system = InjuryPredictionSystem("trained_model.pkl")
    
    # Get valid teams
    valid_teams = system.get_teams()
    print(f"Loaded {len(valid_teams)} teams.")

except Exception as e:
    # Fallback for debugging if files are missing
    print(f"Error loading system: {e}")
    messagebox.showerror("Fatal Error", f"Could not start system.\n\nError: {e}")
    exit()


def load_rosters():
    """Fetch players using system.get_team_roster() and populate lists."""
    name_a = combo_team_a.get()
    name_b = combo_team_b.get()

    if not name_a or not name_b:
        messagebox.showwarning("Missing Info", "Please select both teams first.")
        return

    # Clear current lists
    listbox_a.delete(0, tk.END)
    listbox_b.delete(0, tk.END)

    # Get rosters from backend (Returns list of dicts: [{'name':..., 'stats':...}, ...])
    players_a = system.get_team_roster(name_a)
    players_b = system.get_team_roster(name_b)

    # Populate Listbox A (Home)
    if players_a:
        for p in players_a:
            listbox_a.insert(tk.END, p['name'])
    else:
        listbox_a.insert(tk.END, "No roster data found.")

    # Populate Listbox B (Away)
    if players_b:
        for p in players_b:
            listbox_b.insert(tk.END, p['name'])
    else:
        listbox_b.insert(tk.END, "No roster data found.")

    print(f"Loaded rosters for {name_a} and {name_b}")

def run_prediction():
    """
    1. Runs prediction with NO injuries (Baseline).
    2. Runs prediction WITH selected injuries.
    3. Calculates the 'Impact' (Difference).
    """
    team_a = combo_team_a.get()
    team_b = combo_team_b.get()
    
    if not team_a or not team_b:
        messagebox.showwarning("Error", "Please load rosters and select teams first.")
        return
    
    # Get user selections (The Injured Players)
    # curselection() returns indices, we map them back to names
    injured_a = [listbox_a.get(i) for i in listbox_a.curselection()]
    injured_b = [listbox_b.get(i) for i in listbox_b.curselection()]

    try:
 
        baseline_result = system.predict_matchup(team_a, team_b, injured_a=[], injured_b=[])
        base_prob_a = baseline_result['team_a_win_probability']
        
        
        
        actual_result = system.predict_matchup(team_a, team_b, injured_a=injured_a, injured_b=injured_b)
        actual_prob_a = actual_result['team_a_win_probability']
        
        # Calculate impact
        diff = actual_prob_a - base_prob_a
        winner = actual_result['predicted_winner']
        conf = actual_result['confidence'] * 100

        # formatting the impact string
        if diff < 0:
            impact_msg = f"Injuries hurt {team_a} by {abs(diff)*100:.2f}%"
        elif diff > 0:
            impact_msg = f" {team_a} chances improved by {diff*100:.2f}% (relative to opponent)"
        else:
            impact_msg = "No statistical impact."

        # Prepare the message
        result_msg = (
            f"PREDICTED WINNER: {winner}\n"
            f"Confidence: {conf:.1f}%\n"
            f"{'-'*30}\n"
            f"ANALYSIS:\n"
            f"Full Health Win Chance ({team_a}): {base_prob_a*100:.1f}%\n"
            f"With Injuries Selected, Win Chance ({team_a}): {actual_prob_a*100:.1f}%\n"
            f"\nIMPACT: {impact_msg}"
        )

        messagebox.showinfo("Prediction Analysis", result_msg)
        
        # Optional: Print to console for detailed logs
        system.print_prediction(actual_result)

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


window = tk.Tk()
window.title("NBA Matchup Analyzer")
window.geometry("700x650")
# Header
tk.Label(window, text="NBA Matchup Analyzer", font=("Arial", 16, "bold")).pack(pady=10)
tk.Label(window, text="Select players in the lists below to mark them as INJURED", font=("Arial", 10, "italic"), fg="gray").pack()

# Team Selection
frame_top = tk.Frame(window)
frame_top.pack(pady=15)

tk.Label(frame_top, text="Home Team:").grid(row=0, column=0, padx=5)
combo_team_a = ttk.Combobox(frame_top, values=valid_teams, state="readonly", width=22)
combo_team_a.grid(row=0, column=1, padx=5)
if len(valid_teams) > 0: combo_team_a.current(0)
tk.Label(frame_top, text="Away Team:").grid(row=0, column=2, padx=5)
combo_team_b = ttk.Combobox(frame_top, values=valid_teams, state="readonly", width=22)
combo_team_b.grid(row=0, column=3, padx=5)
if len(valid_teams) > 1: combo_team_b.current(1)

# Load Button
btn_load = tk.Button(window, text="LOAD ROSTERS", command=load_rosters, bg="#dddddd", width=20)
btn_load.pack(pady=5)

# Rosters Area
frame_lists = tk.Frame(window)
frame_lists.pack(fill="both", expand=True, padx=20, pady=10)

# Left List (Home)
frame_a = tk.Frame(frame_lists)
frame_a.pack(side="left", fill="both", expand=True, padx=5)
tk.Label(frame_a, text="Home Roster (Select Injured)", font=("Arial", 10, "bold")).pack()
scroll_a = tk.Scrollbar(frame_a)
scroll_a.pack(side="right", fill="y")
listbox_a = tk.Listbox(frame_a, selectmode=tk.MULTIPLE, height=18, width=30, yscrollcommand=scroll_a.set)
listbox_a.pack(side="left", fill="both", expand=True)
scroll_a.config(command=listbox_a.yview)

# Right List (Away)
frame_b = tk.Frame(frame_lists)
frame_b.pack(side="right", fill="both", expand=True, padx=5)
tk.Label(frame_b, text="Away Roster (Select Injured)", font=("Arial", 10, "bold")).pack()
scroll_b = tk.Scrollbar(frame_b)
scroll_b.pack(side="right", fill="y")
listbox_b = tk.Listbox(frame_b, selectmode=tk.MULTIPLE, height=18, width=30, yscrollcommand=scroll_b.set)
listbox_b.pack(side="left", fill="both", expand=True)
scroll_b.config(command=listbox_b.yview)

# Predict Button
btn_predict = tk.Button(window, text="ANALYZE IMPACT", command=run_prediction, bg="green", fg="white", font=("Arial", 12, "bold"), height=2, width=30)
btn_predict.pack(pady=20)

# Run Loop
window.mainloop()

