import pandas as pd
import random
from collections import defaultdict

# -----------------------
# CONFIG
# -----------------------
N_FIXED = 6   # total items per fixed list
N_RANDOM = 8  # total items per random list

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
phase1_df = pd.read_csv("phase1.csv")
phase2_df = pd.read_csv("phase2dpo_augmented.csv")

# Convert to set of (qid, label) pairs for uniqueness
phase1_pairs = set(zip(phase1_df.qid, phase1_df.label))
phase2_pairs = set(zip(phase2_df.qid, phase2_df.label))

# -----------------------
# FILTER EXCLUSIONS
# -----------------------
phase1_filtered = list(phase1_pairs - EXCLUDE_PHASE1 - EXCLUDE_PHASE2)
phase2_filtered = list(phase2_pairs - EXCLUDE_PHASE2)

# -----------------------
# FUNCTION TO SAMPLE BALANCED LIST WITHOUT DUPLICATES
# -----------------------
def sample_balanced_no_dupes(pairs, n_total):
    """
    Sample n_total items with equal 0s and 1s,
    ensuring no duplicates within the list.
    """
    label_groups = defaultdict(list)
    seen_qids = set()
    for qid, label in pairs:
        if qid not in seen_qids:
            label_groups[label].append((qid, label))
            seen_qids.add(qid)
    
    n_each = n_total // 2
    sampled = []
    for label in [0, 1]:
        if len(label_groups[label]) < n_each:
            raise ValueError(f"Not enough items with label {label} to sample {n_each}")
        sampled.extend(random.sample(label_groups[label], n_each))
    
    return sampled

# -----------------------
# SAMPLE FIXED AND RANDOM ITEMS
# -----------------------
fixed_phase1 = sample_balanced_no_dupes(phase1_filtered, N_FIXED)
remaining_phase1 = [p for p in phase1_filtered if p not in fixed_phase1]
random_phase1 = sample_balanced_no_dupes(remaining_phase1, N_RANDOM)

fixed_phase2 = sample_balanced_no_dupes(phase2_filtered, N_FIXED)
remaining_phase2 = [p for p in phase2_filtered if p not in fixed_phase2]
random_phase2 = sample_balanced_no_dupes(remaining_phase2, N_RANDOM)

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
