"""
Randomly drop a percentage of datapoints from the Lorenz attractor dataset.

HOW TO USE:
Just change the DROP_PERCENT value below to whatever percentage you want
to drop (e.g. 10, 20, 33.3, 50), then run this script.

The output file name is generated automatically, e.g.:
    DROP_PERCENT = 10  ->  Lorentz_attractor_dropped_10pct.csv
    DROP_PERCENT = 25  ->  Lorentz_attractor_dropped_25pct.csv
"""

import pandas as pd
import numpy as np

# ============================================================
# SETTINGS — change these values as needed
# ============================================================
INPUT_FILE = "Lorentz_attractor_5s_0.010_rate.csv"
DROP_PERCENT = 2      # <-- percentage of rows to randomly drop (0-100)
RANDOM_SEED = 42       # <-- keep fixed for reproducible results, or set to None for a new random drop each run
# ============================================================


def main():
    # 1. Load the original dataset
    df = pd.read_csv(INPUT_FILE)
    total_rows = len(df)

    # 2. Work out how many rows to drop
    rows_to_drop = int(round(total_rows * DROP_PERCENT / 100))

    # 3. Randomly pick which rows to drop
    rng = np.random.default_rng(RANDOM_SEED)
    drop_indices = rng.choice(df.index, size=rows_to_drop, replace=False)

    # 4. Remove those rows
    df_dropped = df.drop(index=drop_indices).reset_index(drop=True)

    # 5. Build output filename automatically, e.g. Lorentz_attractor_dropped_10pct.csv
    percent_label = str(DROP_PERCENT).rstrip("0").rstrip(".") if "." in str(DROP_PERCENT) else str(DROP_PERCENT)
    output_file = f"Lorentz_attractor_dropped_{percent_label}pct.csv"

    # 6. Save the new dataset
    df_dropped.to_csv(output_file, index=False)

    # 7. Print a short summary
    print(f"Original rows : {total_rows}")
    print(f"Dropped rows  : {rows_to_drop} ({DROP_PERCENT}%)")
    print(f"Remaining rows: {len(df_dropped)}")
    print(f"Saved to      : {output_file}")


if __name__ == "__main__":
    main()