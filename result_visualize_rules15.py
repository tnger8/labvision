import pandas as pd
import ast
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Load Config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

rules = config["rules"]["eating_violation"]
time_bin_minutes = 15  # fixed 15-min interval for this dashboard

# --- Load Data ---
df = pd.read_csv("aggregated_violations.csv")

# Convert timestamps
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

# Convert objects_seen string to list
df['objects_seen'] = df['objects_seen'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# --- Compute duration and true violation ---
df['duration_sec'] = (df['end_time'] - df['start_time']).dt.total_seconds()
df['is_true_violation'] = (df['duration_sec'] >= rules['min_duration_sec'])

# --- Aggregate per 15-min interval ---
df['time_bin'] = df['start_time'].dt.floor(f'{time_bin_minutes}min')
violations_over_time = df.groupby('time_bin')['is_true_violation'].sum().reset_index()
people_over_time = df.groupby('time_bin')['people_detected'].max().reset_index()  # max people in interval

# --- Daily summary ---
daily_summary = df.groupby(df['start_time'].dt.date)['is_true_violation'].sum().reset_index()
daily_summary.rename(columns={'start_time': 'date', 'is_true_violation': 'true_violations'}, inplace=True)

# --- Top objects ---
all_objects = df['objects_seen'].explode()
all_objects = all_objects[all_objects.notna() & (all_objects != '')]
object_counts = all_objects.value_counts().reset_index()
object_counts.columns = ['object', 'count']

# --- Heatmap (hourly bins for clarity) ---
df['hour'] = df['start_time'].dt.floor('60min')
df['day'] = df['start_time'].dt.date
heatmap_data = (
    df[df['is_true_violation']]
    .groupby(['day', 'hour'])['is_true_violation']
    .sum()
    .reset_index()
)
heatmap_data['hour_of_day'] = heatmap_data['hour'].dt.hour
hours_sorted = sorted(heatmap_data['hour_of_day'].unique())
days_sorted = sorted(heatmap_data['day'].unique())
heatmap_matrix = (
    heatmap_data.pivot_table(
        index='hour_of_day',
        columns='day',
        values='is_true_violation',
        fill_value=0
    )
    .reindex(index=hours_sorted, columns=days_sorted, fill_value=0)
)

# --- Session summary with actual durations ---
true_sessions = df[df['is_true_violation']].copy()
true_sessions['duration_sec'] = (true_sessions['end_time'] - true_sessions['start_time']).dt.total_seconds()

# --- Create 2x3 Dashboard ---
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "Violations Over Time", "Daily Violations", "Top Detected Objects",
        "Violation Heatmap", "People vs Violations", "Session Summary"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
        [{"type": "heatmap"}, {"type": "scatter"}, {"type": "table"}]
    ],
    horizontal_spacing=0.12, vertical_spacing=0.15
)

# 1. Violations Over Time
fig.add_trace(
    go.Scatter(
        x=violations_over_time['time_bin'], y=violations_over_time['is_true_violation'],
        mode='lines+markers', name='Violations', line=dict(color='red')
    ),
    row=1, col=1
)

# 2. Daily Violations
fig.add_trace(
    go.Bar(
        x=daily_summary['date'], y=daily_summary['true_violations'], name='Daily Violations'
    ),
    row=1, col=2
)

# 3. Top Objects
topn = object_counts.head(20)
fig.add_trace(
    go.Bar(x=topn['object'], y=topn['count'], name='Top Objects'),
    row=1, col=3
)

# 4. Heatmap
fig.add_trace(
    go.Heatmap(
        z=heatmap_matrix.values,
        x=[str(d) for d in heatmap_matrix.columns],
        y=[f"{h:02d}:00" for h in heatmap_matrix.index],
        colorscale='Reds',
        showscale=False  # remove legend/colorbar
    ),
    row=2, col=1
)

# 5. People vs Violations
fig.add_trace(
    go.Bar(
        x=people_over_time['time_bin'], y=people_over_time['people_detected'],
        name='People', marker_color='blue'
    ),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(
        x=violations_over_time['time_bin'], y=violations_over_time['is_true_violation'],
        mode='lines+markers', name='Violations', line=dict(color='red')
    ),
    row=2, col=2
)

# 6. Session Summary Table
if not true_sessions.empty:
    fig.add_trace(
        go.Table(
            header=dict(values=["Start", "End", "Duration (s)", "People", "Objects"]),
            cells=dict(values=[
                true_sessions['start_time'].astype(str),
                true_sessions['end_time'].astype(str),
                true_sessions['duration_sec'].astype(int),
                true_sessions['people_detected'],
                true_sessions['objects_seen'].astype(str)
            ])
        ),
        row=2, col=3
    )

# --- Layout ---
fig.update_layout(
    height=950, width=1400,
    title_text="LabVision: 15-min Interval True Violations Dashboard",
    template='plotly_white',
    showlegend=True
)

fig.show()
