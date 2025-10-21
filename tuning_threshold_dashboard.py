import os
import cv2
import glob
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.io as pio

# =============================
# CONFIGURATION
# =============================
TEST_VIDEO_FOLDER = "test_videos"
GROUND_TRUTH_CSV = os.path.join(TEST_VIDEO_FOLDER, "ground_truth_all.csv")
THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.8]   # you can expand this list
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

POSE_MODEL_PATH = "yolov8n-pose.pt"

# =============================
# HELPER FUNCTIONS
# =============================

def run_inference(video_path, output_csv, threshold):
    """Run inference with a given threshold and save results as CSV."""
    model = YOLO(POSE_MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, verbose=False)
        keypoints = results[0].keypoints
        action = "none"

        if keypoints is not None and len(keypoints.xy) > 0:
            # Simulate detection: threshold controls probability of "eating"
            prob = np.random.random()
            if prob > (1 - threshold):  
                action = "eating"

        rows.append([frame_count, action, 1])  # video_id=1

    cap.release()

    df = pd.DataFrame(rows, columns=["frame", "action", "video_id"])
    df.to_csv(output_csv, index=False)
    print(f"Saved inference CSV: {output_csv}")


def compute_metrics(gt_df, pred_df):
    """Compute metrics (frame-level) between ground truth and predictions."""
    gt_df["video_id"] = gt_df["video_id"].astype(str)
    pred_df["video_id"] = pred_df["video_id"].astype(str)

    data = pd.merge(gt_df, pred_df, on=["frame", "video_id"], how="inner", suffixes=('_gt', '_pred'))
    data["gt_binary"] = data["action_gt"].apply(lambda x: 1 if str(x).lower() in ["eating", "drink", "drinking"] else 0)
    data["pred_binary"] = data["action_pred"].apply(lambda x: 1 if str(x).lower() in ["eating", "drink", "drinking"] else 0)

    acc = accuracy_score(data["gt_binary"], data["pred_binary"])
    prec = precision_score(data["gt_binary"], data["pred_binary"], zero_division=0)
    rec = recall_score(data["gt_binary"], data["pred_binary"], zero_division=0)
    f1 = f1_score(data["gt_binary"], data["pred_binary"], zero_division=0)

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}, data


def generate_dashboard(metrics_df, output_html):
    """Generate interactive HTML dashboard for threshold tuning."""
    fig = go.Figure()

    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        fig.add_trace(go.Scatter(
            x=metrics_df["Threshold"], y=metrics_df[metric],
            mode="lines+markers", name=metric
        ))

    fig.update_layout(
        title="Threshold Tuning Dashboard",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )

    pio.write_html(fig, file=output_html, auto_open=True)
    print(f"✅ Dashboard saved to: {output_html}")


# =============================
# MAIN PIPELINE
# =============================
def main():
    gt_all = pd.read_csv(GROUND_TRUTH_CSV)
    all_metrics = []

    for thresh in THRESHOLDS:
        print(f"\n=== Threshold: {thresh} ===")
        out_dir = os.path.join(RESULTS_DIR, f"thresh_{thresh}")
        os.makedirs(out_dir, exist_ok=True)

        pred_dfs = []
        video_files = sorted(glob.glob(os.path.join(TEST_VIDEO_FOLDER, "*.mp4")))

        # --- Run inference per video ---
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_csv = os.path.join(out_dir, f"{video_name}.csv")

            run_inference(video_path, output_csv, threshold=thresh)
            pred_dfs.append(pd.read_csv(output_csv))

        # --- Combine all inferences for this threshold ---
        if pred_dfs:
            pred_all = pd.concat(pred_dfs, ignore_index=True)
            pred_all.to_csv(os.path.join(out_dir, f"inference_all_thresh_{thresh}.csv"), index=False)
            print(f"✅ Combined inference saved: {out_dir}/inference_all_thresh_{thresh}.csv")
        else:
            print(f"⚠️ No inference data found for threshold {thresh}")
            continue

        # --- Compute metrics ---
        metrics, merged_data = compute_metrics(gt_all, pred_all)
        metrics["Threshold"] = thresh
        all_metrics.append(metrics)

    # --- Create summary dashboard ---
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        output_html = os.path.join(RESULTS_DIR, "threshold_dashboard.html")
        generate_dashboard(metrics_df, output_html)
    else:
        print("⚠️ No metrics to display — check data paths or inference results.")


if __name__ == "__main__":
    main()
