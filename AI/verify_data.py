import pandas as pd

df = pd.read_csv("synthetic_workouts.csv")

print("=== First 5 rows ===")
print(df.head().to_string())
print()

print(f"=== Dataset shape: {df.shape} ===")
print(f"=== Features: {list(df.columns)} ===")
print()

print("=== Edge case: ultra-short bursts (<12 min, high HR) ===")
short_intense = df[(df["duration_mins"] < 12) & (df["avg_hr"] > 140)]
print(f"  Found {len(short_intense)} samples")
print(f"  All labeled 0 (Low)? {(short_intense['effectiveness_label'] == 0).all()}")
print()

print("=== Edge case: short heavy singles (<10 min, EMG >500) ===")
singles = df[(df["duration_mins"] < 10) & (df["avg_emg"] > 500)]
print(f"  Found {len(singles)} samples")
print(f"  All labeled 0 (Low)? {(singles['effectiveness_label'] == 0).all()}")
print()

print("=== Edge case: powerlifting (low HR <105, high EMG >530, >35 min) ===")
power = df[(df["avg_hr"] < 105) & (df["avg_emg"] > 530) & (df["duration_mins"] > 35)]
print(f"  Found {len(power)} samples")
labels = dict(power["effectiveness_label"].value_counts())
print(f"  Label distribution: {labels}")
print()

print("=== Edge case: phone scrolling (>40 min, EMG <180, low zone >65%) ===")
scroll = df[(df["duration_mins"] > 40) & (df["avg_emg"] < 180) & (df["pct_time_low"] > 65)]
print(f"  Found {len(scroll)} samples")
labels = dict(scroll["effectiveness_label"].value_counts())
print(f"  Label distribution: {labels}")
print()

print("=== Edge case: high-rep endurance (>180 reps, >33 min) ===")
endurance = df[(df["total_reps"] > 180) & (df["duration_mins"] > 33)]
print(f"  Found {len(endurance)} samples")
labels = dict(endurance["effectiveness_label"].value_counts())
print(f"  Label distribution: {labels}")
print()

print("=== Verify avg_reps_set is NOT in the dataset ===")
print(f"  'avg_reps_set' in columns? {'avg_reps_set' in df.columns}")
