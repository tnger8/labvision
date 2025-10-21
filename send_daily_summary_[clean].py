# send_daily_summary.py
# Generates dashboard image + text summary from aggregated_violations.csv
# Sends report via Gmail SMTP (App Password) and Telegram.
#
# Requirements:
#   pip install pandas plotly kaleido requests

import pandas as pd
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
import os
from datetime import datetime

# -----------------------
# CONFIG - EDIT THESE
# -----------------------
CSV_PATH = "aggregated_violations.csv"

# Email (Gmail) config - use App Password (no spaces)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "{your_own_email}"
EMAIL_PASSWORD = "{yourownpasscode}"  # App password from Google

EMAIL_TO = ["{your_email}@gmail.com"]  # list of recipients

# Telegram config
TELEGRAM_TOKEN = 'TELEGRAM_TOKEN_to_edit'
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID_edit'

# Output files
OUTPUT_PNG = "daily_summary.png"
OUTPUT_HTML = "daily_summary.html"

# Food objects considered violations (tweak as needed)
FOOD_OBJECTS = ['bottle', 'cup', 'bowl', 'banana', 'donut', 'apple', 'hot dog', 'sandwich', 'orange', 'carrot']

# Small gap tolerance (seconds) when grouping frames into sessions
SESSION_GAP_TOLERANCE = 30

# -----------------------
# Helper functions
# -----------------------
def safe_literal_eval(obj_str):
    try:
        return ast.literal_eval(obj_str) if isinstance(obj_str, str) and obj_str.strip() != "" else []
    except Exception:
        return []

def send_email_with_image(subject, body_text, image_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = ", ".join(EMAIL_TO)
    msg['Subject'] = subject

    msg.attach(MIMEText(body_text, 'plain'))

    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read(), name=os.path.basename(image_path))
        msg.attach(img)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)

def send_telegram_with_image(caption, image_path):
    # send text first (useful if image send fails)
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": caption})
    except Exception as e:
        print("Warning: Telegram text send failed:", e)

    # send image
    with open(image_path, "rb") as img:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": img},
            timeout=30
        )
    return resp

# -----------------------
# Load & preprocess
# -----------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Attempt to parse and normalize columns, be robust to different CSV schemas
# Expected columns (some may be missing depending on pipeline): start_time, end_time, frames_count, objects_seen, violation
if 'start_time' in df.columns:
    df['start_time'] = pd.to_datetime(df['start_time'], dayfirst=True, errors='coerce')
else:
    # fallback: try a generic timestamp column
    possible_ts = [c for c in df.columns if 'time' in c.lower()]
    if possible_ts:
        df['start_time'] = pd.to_datetime(df[possible_ts[0]], errors='coerce')
    else:
        raise ValueError("No start_time (or similar) column found in CSV.")

if 'end_time' in df.columns:
    df['end_time'] = pd.to_datetime(df['end_time'], dayfirst=True, errors='coerce')
else:
    df['end_time'] = df['start_time']

if 'frames_count' not in df.columns:
    df['frames_count'] = 1  # fallback assume 1 frame rows

# parse objects_seen to list
if 'objects_seen' in df.columns:
    df['objects_seen'] = df['objects_seen'].apply(safe_literal_eval)
else:
    df['objects_seen'] = [[] for _ in range(len(df))]

# define violation flag if not present using FOOD_OBJECTS
if 'violation' not in df.columns:
    df['violation'] = df['objects_seen'].apply(lambda objs: any(item in FOOD_OBJECTS for item in objs))

# drop rows with invalid start_time
df = df[df['start_time'].notna()].copy()
if df.empty:
    raise ValueError("No valid timestamp rows in CSV after parsing.")

# -----------------------
# Aggregate sessions (group consecutive food-detected rows)
# -----------------------
df_food = df[df['violation']].copy().sort_values('start_time')

sessions = []
current = None
for idx, row in df_food.iterrows():
    st = row['start_time']
    et = row['end_time'] if pd.notna(row['end_time']) else st
    fc = int(row.get('frames_count', 1))
    objs = set(row.get('objects_seen', []))
    if current is None:
        current = {'start_time': st, 'end_time': et, 'frames_count': fc, 'objects_seen': objs}
    else:
        gap = (st - current['end_time']).total_seconds()
        if gap <= SESSION_GAP_TOLERANCE:
            # extend
            current['end_time'] = max(current['end_time'], et)
            current['frames_count'] += fc
            current['objects_seen'].update(objs)
        else:
            sessions.append(current)
            current = {'start_time': st, 'end_time': et, 'frames_count': fc, 'objects_seen': objs}
if current:
    sessions.append(current)

session_df = pd.DataFrame(sessions)
if not session_df.empty:
    session_df['objects_seen'] = session_df['objects_seen'].apply(lambda s: list(s))

# -----------------------
# Build dashboard (2x2)
# -----------------------
# Daily summary
daily_summary = df.groupby(df['start_time'].dt.date).agg(
    total_frames=('frames_count', 'sum'),
    total_violations=('violation', 'sum')
).reset_index().rename(columns={'start_time': 'date'})

# Violations over time (group by minute)
violations_over_time = df.set_index('start_time').resample('1T')['violation'].sum().reset_index()

# Top objects
all_objects = df['objects_seen'].explode().dropna()
object_counts = all_objects.value_counts().reset_index()
object_counts.columns = ['object', 'count']

# Heatmap (5-min bins)
df['time_bin'] = df['start_time'].dt.floor('5T')
df['hour'] = df['time_bin'].dt.hour
df['day'] = df['time_bin'].dt.date
heatmap_data = df.groupby(['day', 'hour'])['violation'].sum().reset_index()
heatmap_matrix = heatmap_data.pivot(index='hour', columns='day', values='violation').fillna(0)

# Create figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Violations Over Time (1-min bins)", "Daily Violations Summary", "Top Detected Objects", "Violation Heatmap"),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "heatmap"}]],
    horizontal_spacing=0.12, vertical_spacing=0.15
)

# 1: Violations over time
fig.add_trace(
    go.Scatter(
        x=violations_over_time['start_time'],
        y=violations_over_time['violation'],
        mode='lines+markers',
        name='Violations'
    ),
    row=1, col=1
)

# overlay sessions as vrects
for idx, s in (session_df.sort_values('start_time').iterrows() if not session_df.empty else []):
    srow = s if not isinstance(s, tuple) else s[1]  # in case iterrows returns tuple
# Add vrects via loop (Plotly expects figs to use add_vrect)
if not session_df.empty:
    for _, srow in session_df.iterrows():
        fig.add_vrect(
            x0=srow['start_time'],
            x1=srow['end_time'],
            fillcolor="LightSalmon",
            opacity=0.3,
            layer="below",
            row=1, col=1,
            line_width=0
        )

# 2: Daily summary
fig.add_trace(
    go.Bar(x=daily_summary['start_time'] if 'start_time' in daily_summary.columns else daily_summary['date'],
           y=daily_summary['total_violations'],
           name='Daily Violations'),
    row=1, col=2
)

# 3: Top objects (limit to top 20 for readability)
topn = object_counts.head(20)
fig.add_trace(
    go.Bar(x=topn['object'], y=topn['count'], name='Top Objects'),
    row=2, col=1
)

# 4: Heatmap
fig.add_trace(
    go.Heatmap(
        z=heatmap_matrix.values,
        x=[str(d) for d in heatmap_matrix.columns],
        y=heatmap_matrix.index,
        colorscale='Reds',
        colorbar=dict(title="Violations")
    ),
    row=2, col=2
)

fig.update_layout(height=900, width=1200, title_text="LabVision Daily Summary", showlegend=False)
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text="Violations", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Violations", row=1, col=2)
fig.update_xaxes(title_text="Object", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=2)
fig.update_yaxes(title_text="Hour of Day", row=2, col=2)

# Export dashboard
try:
    fig.write_html(OUTPUT_HTML)
    # write_image requires kaleido
    fig.write_image(OUTPUT_PNG)
    print(f"✅ Dashboard exported to {OUTPUT_HTML} and {OUTPUT_PNG}")
except Exception as e:
    print("⚠️ Failed to export figure image (kaleido may be missing). Error:", e)
    # still try to save HTML only
    fig.write_html(OUTPUT_HTML)
    print(f"✅ Dashboard HTML exported to {OUTPUT_HTML}")

# -----------------------
# Build summary text
# -----------------------
total_frames = int(df['frames_count'].sum())
total_violations = int(df['violation'].sum())
top_objects = object_counts.head(3)['object'].tolist()
num_sessions = len(session_df)
longest_session_frames = int(session_df['frames_count'].max()) if not session_df.empty else 0
longest_session_minutes = (session_df['end_time'] - session_df['start_time']).dt.total_seconds().max() / 60 if not session_df.empty else 0

latest_day = df['start_time'].dt.date.max()

summary_text = (
    f"📊 LabVision Daily Summary ({latest_day})\n"
    f"----------------------------------------\n"
    f"Total frames processed: {total_frames}\n"
    f"Total violations detected: {total_violations}\n"
    f"Top objects: {', '.join(top_objects) if top_objects else 'N/A'}\n"
    f"Eating sessions: {num_sessions}\n"
    f"Longest session: {longest_session_frames} frames (~{longest_session_minutes:.1f} min)\n\n"
    f"Attached: dashboard image ({OUTPUT_PNG})\n"
)
print(summary_text)

# -----------------------
# Send email + telegram
# -----------------------
subject = f"LabVision Daily Report - {latest_day}"

# Send email (with fallback to Telegram)
email_ok = False
try:
    send_email_with_image(subject, summary_text, OUTPUT_PNG)
    print("✅ Email sent successfully.")
    email_ok = True
except Exception as e:
    print("❌ Email failed:", e)

# Send Telegram regardless (and also as fallback)
try:
    resp = send_telegram_with_image(summary_text, OUTPUT_PNG)
    if resp is not None and resp.status_code == 200:
        print("✅ Telegram summary sent.")
    else:
        print("❌ Telegram send may have failed. Response:", resp.status_code if resp is not None else None, resp.text if resp is not None else "")
except Exception as e:
    print("❌ Telegram failed:", e)

print("✅ send_daily_summary.py finished.")
