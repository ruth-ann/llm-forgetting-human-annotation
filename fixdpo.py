import pandas as pd

# Load files
phase1 = pd.read_csv("phase1.csv")
phase2dpo = pd.read_csv("phase2dpo.csv")

# qids already in phase2dpo
p2_qids = set(phase2dpo["qid"].unique())

# Select all rows from phase1 where label is 0 and qid not in phase2dpo
label0_not_in_phase2 = phase1[(phase1["label"] == 0) & (~phase1["qid"].isin(p2_qids))]

# Combine with existing phase2dpo
augmented = pd.concat([phase2dpo, label0_not_in_phase2], ignore_index=True)

# Save the result
augmented.to_csv("phase2dpo_augmented.csv", index=False)

print("done")
