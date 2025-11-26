import pandas as pd
import random

# -----------------------
# CONFIG
# -----------------------
N_FIXED = 6
N_RANDOM = 12

# -----------------------
# EXCLUSIONS
# -----------------------
EXCLUDE_PHASE1 = {
    (120, 0), (120, 1),
    (137, 1), (137, 0),
    (232, 0), (232, 1),
    (2155, 1), (2155, 0),
}

EXCLUDE_PHASE2 = {
    (1429, 0), (1429, 1),
    (1609, 0), (1609, 1),
    (2039, 0), (2039, 1),
    (3379, 0), (3379, 1),
}

# -----------------------
# LOAD DATA
# -----------------------
phase1_df = pd.read_csv("tmp.csv")  # Phase 1
phase2_df = pd.read_csv("dpo.csv")  # Phase 2

# Convert to list of (qid, label) pairs
phase1_pairs = list(zip(phase1_df.qid, phase1_df.label))
phase2_pairs = list(zip(phase2_df.qid, phase2_df.label))

# -----------------------
# FILTER EXCLUSIONS
# -----------------------
phase1_filtered = [p for p in phase1_pairs if p not in EXCLUDE_PHASE1]
phase2_filtered = [p for p in phase2_pairs if p not in EXCLUDE_PHASE2]

# -----------------------
# SAMPLE FIXED AND RANDOM ITEMS
# -----------------------
fixed_phase1 = random.sample(phase1_filtered, N_FIXED)
random_phase1 = random.sample([p for p in phase1_filtered if p not in fixed_phase1], N_RANDOM)

fixed_phase2 = random.sample(phase2_filtered, N_FIXED)
random_phase2 = random.sample([p for p in phase2_filtered if p not in fixed_phase2], N_RANDOM)

# -----------------------
# PRINT RESULTS
# -----------------------
def print_pairs(name, pairs):
    print(f"\n{name} = [")
    for qid, label in pairs:
        print(f"    ({qid}, {label}),")
    print("]")

print_pairs("FIXED_PHASE1", fixed_phase1)
print_pairs("RANDOM_PHASE1", random_phase1)
print_pairs("FIXED_PHASE2", fixed_phase2)
print_pairs("RANDOM_PHASE2", random_phase2)
