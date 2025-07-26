# run_evaluation.py

import os
import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score, r2_score
from custom_definitions import * 
# --- Main Competition Configuration ---
# This dictionary controls the entire script.
# It defines the task and metric for each day.
ALL_DAYS_INFO = {
    1: {'task': 'classification', 'metric': 'Accuracy'},
    2: {'task': 'regression', 'metric': 'R2-Score'},
    3: {'task': 'classification', 'metric': 'Accuracy'},
    4: {'task': 'regression', 'metric': 'R2-Score'}
}

def get_rank_points(rank):
    """Assigns points based on rank."""
    if rank == 1: return 100
    if rank == 2: return 90
    if rank == 3: return 80
    if rank == 4: return 75
    if rank == 5: return 70
    if 6 <= rank <= 10: return 60
    return 25

def validate_day(day_num, task_type):
    """
    Validates all submissions for a specific day.
    This function is model-agnostic; it just loads and runs the .pkl file.
    """
    print(f"--- Starting Validation for Day {day_num} ({task_type}) ---")
    models_dir = f"day{day_num}_submissions"
    validation_dir = f"day{day_num}_validation"
    output_scores_path = f"day{day_num}_scores.csv"

    if not os.path.exists(models_dir) or not os.path.exists(validation_dir):
        print(f"Error: Required directories for Day {day_num} not found. Aborting.")
        return

    try:
        X_val = pd.read_csv(os.path.join(validation_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(validation_dir, 'y_val.csv')).squeeze()
    except FileNotFoundError as e:
        print(f"Error loading validation data: {e}. Aborting.")
        return

    daily_results = []
    for file_name in os.listdir(models_dir):
        if file_name.endswith(".pkl"):
            participant_name = os.path.splitext(file_name)[0]
            model_path = os.path.join(models_dir, file_name)
            score = -999.0  # Default score for failed models

            try:

                pipeline = joblib.load(model_path)
                predictions = pipeline.predict(X_val)

                if task_type == 'classification':
                    score = accuracy_score(y_val, predictions)
                elif task_type == 'regression':
                    score = r2_score(y_val, predictions)
                
                print(f"  [SUCCESS] Evaluated: {participant_name:<20} | Score: {score:.4f}")

            except Exception as e:
                print(f"  [ FAILED] Evaluated: {participant_name:<20} | Reason: {e}")
            
            daily_results.append({
                "Participant": participant_name,
                ALL_DAYS_INFO[day_num]['metric']: score
            })

    if daily_results:
        scores_df = pd.DataFrame(daily_results)
        scores_df.to_csv(output_scores_path, index=False)
        print(f"\nDay {day_num} scores saved to {output_scores_path}")

def update_leaderboard():
    """
    Recalculates the entire leaderboard from all existing daily score CSVs.
    """
    print("\n--- Updating Main Leaderboard ---")
    master_leaderboard = pd.DataFrame()

    for day_num, info in ALL_DAYS_INFO.items():
        scores_file = f"day{day_num}_scores.csv"
        if os.path.exists(scores_file):
            print(f"Processing scores from {scores_file}...")
            daily_df = pd.read_csv(scores_file)
            metric_col = info['metric']
            
            daily_df['Rank'] = daily_df[metric_col].rank(method='min', ascending=False)
            daily_df[f'Day_{day_num}_Points'] = daily_df['Rank'].apply(get_rank_points)
            
            daily_summary = daily_df[['Participant', f'Day_{day_num}_Points']]

            if master_leaderboard.empty:
                master_leaderboard = daily_summary
            else:
                master_leaderboard = pd.merge(master_leaderboard, daily_summary, on='Participant', how='outer')

    if master_leaderboard.empty:
        print("No score files found. Leaderboard not updated.")
        return

    point_columns = [col for col in master_leaderboard.columns if '_Points' in col]
    master_leaderboard = master_leaderboard.fillna(0)
    master_leaderboard[point_columns] = master_leaderboard[point_columns].astype(int)
    master_leaderboard['Total_Points'] = master_leaderboard[point_columns].sum(axis=1)

    master_leaderboard = master_leaderboard.sort_values(by='Total_Points', ascending=False).reset_index(drop=True)
    master_leaderboard['Overall_Rank'] = master_leaderboard.index + 1

    cols = ['Overall_Rank', 'Participant', 'Total_Points'] + sorted(point_columns)
    master_leaderboard = master_leaderboard[cols]

    master_leaderboard.to_csv('leaderboard.csv', index=False)
    print("\nLeaderboard successfully updated and saved to leaderboard.csv.")
    print("--- CURRENT LEADERBOARD ---")
    print(master_leaderboard.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated evaluation script for the AIRAC Challenge.")
    parser.add_argument("--day", type=int, required=True, help="The challenge day number to validate (1, 2, 3, or 4).")
    args = parser.parse_args()

    day_to_validate = args.day
    if day_to_validate not in ALL_DAYS_INFO:
        print(f"Error: Day {day_to_validate} is not valid. Please choose from {list(ALL_DAYS_INFO.keys())}.")
    else:
        task = ALL_DAYS_INFO[day_to_validate]['task']
        validate_day(day_to_validate, task)
        update_leaderboard()