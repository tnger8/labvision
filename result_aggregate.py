import pandas as pd
import os

# File paths
activity_log_file = "activity_log.csv"        # Your input log
master_log_file = "master_activity_log.csv"  # Maintains all historical data
aggregated_file = "aggregated_violations.csv"

# Load new activity log
df_new = pd.read_csv(activity_log_file)

# Standardize boolean columns
bool_cols = ["hand_near_mouth", "temporal_detect", "violation"]
df_new[bool_cols] = df_new[bool_cols].astype(bool)

# If master log exists, load and append only new rows
if os.path.exists(master_log_file):
    df_master = pd.read_csv(master_log_file)
    # Only append rows that are not already in master (based on timestamp + frame)
    df_new = df_new.merge(df_master[['timestamp', 'frame']], 
                          on=['timestamp', 'frame'], 
                          how='left', 
                          indicator=True)
    df_new = df_new[df_new['_merge'] == 'left_only'].drop(columns=['_merge'])
    df_master = pd.concat([df_master, df_new], ignore_index=True)
else:
    df_master = df_new

# Save updated master log
df_master.to_csv(master_log_file, index=False)

# --- Aggregation ---
# Mark consecutive violations as a single event
df_master = df_master.sort_values(['timestamp', 'frame'])
df_master['violation_group'] = (df_master['violation'] != df_master['violation'].shift()).cumsum()

agg = df_master[df_master["violation"]].groupby("violation_group").agg(
    start_time=('timestamp', 'first'),
    end_time=('timestamp', 'last'),
    frames_count=('frame', 'count'),
    objects_seen=('objects', lambda x: list(set(sum(x.apply(eval), []))))
).reset_index(drop=True)

# Save aggregated results
agg.to_csv(aggregated_file, index=False)
print("Aggregation complete! Saved to", aggregated_file)
