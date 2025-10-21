# summary_utils.py
import pandas as pd
import numpy as np

def generate_summary(df, session_df, true_violations):
    """Generate a natural-language summary for daily report."""
    if df.empty:
        return "No data available to summarize."

    total_people = df['people_detected'].sum() if 'people_detected' in df.columns else None
    total_frames = df['frames_count'].sum() if 'frames_count' in df.columns else None
    total_violations = df['violation'].sum()
    total_true_sessions = len(true_violations)
    total_duration = true_violations['duration_sec'].sum() if not true_violations.empty else 0

    if not df['start_time'].isna().all():
        active_period = df['start_time'].max() - df['start_time'].min()
    else:
        active_period = None

    # Peak violation time
    violation_counts = df.groupby(df['start_time'].dt.floor('5T'))['violation'].sum()
    if not violation_counts.empty:
        peak_time = violation_counts.idxmax().strftime('%H:%M')
        peak_value = violation_counts.max()
    else:
        peak_time, peak_value = "N/A", 0

    summary = "📊 **LabVision Daily Summary**\n"
    summary += f"- Total frames analyzed: {total_frames:,}\n" if total_frames else ""
    summary += f"- Total detected violations: {total_violations}\n"
    summary += f"- True eating sessions: {total_true_sessions}\n"
    summary += f"- Total eating duration: {int(total_duration // 60)} min {int(total_duration % 60)} sec\n"
    summary += f"- Peak violation at: {peak_time} ({peak_value} detections)\n"
    if total_people:
        summary += f"- Total people detected across frames: {total_people}\n"
    if active_period:
        summary += f"- Active monitoring period: {str(active_period)}\n"

    if total_true_sessions == 0:
        summary += "\n✅ No sustained eating activity detected based on configured thresholds."
    elif total_true_sessions == 1:
        summary += "\n⚠️ 1 eating session detected — verify for policy compliance."
    else:
        summary += f"\n🚨 Multiple ({total_true_sessions}) eating sessions detected today."

    return summary
