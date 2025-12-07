import os
import csv
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


def compute_accuracy(csv_files):
    total = 0
    correct = 0

    for file_path in csv_files:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row["true_label"].strip() == row["label"].strip():
                    correct += 1

    return (correct / total) if total else 0.0


def compute_fleiss_kappa(csv_files):
    # qid → list of labels
    by_qid = defaultdict(list)

    for file_path in csv_files:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row["qid"].strip()
                label = row["label"].strip()
                by_qid[qid].append(label)

    # collect all label values seen
    all_labels = sorted({lbl for labels in by_qid.values() for lbl in labels})

    # build matrix: rows = items, columns = labels
    rows = []
    for qid, labels in by_qid.items():
        counts = Counter(labels)
        row = [counts.get(lbl, 0) for lbl in all_labels]
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_labels)

    return fleiss_kappa(df.values)


def main():
    results_dir = Path("results")

    phase1_files = list(results_dir.glob("*phase1.csv"))
    phase2_files = list(results_dir.glob("*phase2.csv"))

    phase1_accuracy = compute_accuracy(phase1_files)
    phase2_accuracy = compute_accuracy(phase2_files)

    phase1_kappa = compute_fleiss_kappa(phase1_files)
    phase2_kappa = compute_fleiss_kappa(phase2_files)

    print("Phase 1 accuracy:", phase1_accuracy)
    print("Phase 1 Fleiss' κ:", phase1_kappa)

    print("Phase 2 accuracy:", phase2_accuracy)
    print("Phase 2 Fleiss' κ:", phase2_kappa)


if __name__ == "__main__":
    main()
