import pandas as pd
import os
import sys
import shutil

def process_csv(respondent_name):
    original_file = os.path.join("results", f"{respondent_name}_responses_phase1.csv")
    backup_file = os.path.join("results", f"{respondent_name}_responses_phase1_old.csv")
    output_file = os.path.join("results", f"{respondent_name}_responses_phase1_max.csv")

    if not os.path.exists(original_file):
        print(f"File not found: {original_file}")
        return

    shutil.copy2(original_file, backup_file)

    df = pd.read_csv(original_file)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Keep only the row with the latest timestamp per qid
    df_max = df.loc[df.groupby("qid")["timestamp"].idxmax()]

    df_max.to_csv(output_file, index=False)
    print(f"Processed {original_file}. Backup: {backup_file}. Output: {output_file}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <respondent_name>")
        sys.exit(1)
    process_csv(sys.argv[1])
