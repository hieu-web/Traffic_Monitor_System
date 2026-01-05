
import os
import cv2
import time
import tempfile
import zipfile
import io
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Environment Fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# UI Colors
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)

@st.cache_resource
def load_ocr_engine():
    """Load PaddleOCR without deprecated arguments for stability."""
    return PaddleOCR(use_angle_cls=True, lang="en")

class TrafficCore:
    def __init__(self):
        # Path logic for your folder structure: models/best.pt
        base_path = Path(__file__).parent
        model_path = base_path / "models" / "best.pt"
        
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            st.warning(f"Custom model not found at {model_path}. Using yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            
        self.reset()

    def reset(self):
        """Reset all counters and session data."""
        self.stats = {"Total": 0, "Violation": 0, "Car": 0, "Motorcycle": 0, "Truck": 0, "Bus": 0}
        self.counted_ids = set()
        self.violated_ids = set()
        self.evidence = []
        self.light_buffer = deque(maxlen=15)
        self.last_violation_time = {}

    def get_light_state(self, frame: np.ndarray, rect: List[int]) -> str:
        """Detect Red, Yellow, or Green state using HSV masking."""
        x, y, w, h = rect
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0: return "OFF"
        
        hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
        h_unit = h // 3
        
        # Split zones: Top (Red), Middle (Yellow), Bottom (Green)
        red_zone = hsv[0:h_unit, :]
        yellow_zone = hsv[h_unit:2*h_unit, :]
        
        r_mask = cv2.bitwise_or(cv2.inRange(red_zone, (0, 150, 100), (10, 255, 255)),
                               cv2.inRange(red_zone, (160, 150, 100), (180, 255, 255)))
        y_mask = cv2.inRange(yellow_zone, (15, 150, 100), (35, 255, 255))
        
        if cv2.countNonZero(r_mask) > (w * h_unit * 0.15): return "RED"
        if cv2.countNonZero(y_mask) > (w * h_unit * 0.15): return "YELLOW"
        return "GREEN"

    def process_frame(self, frame, frame_idx, conf, stop_line_p, safety_offset, light_rect):
        H, W = frame.shape[:2]
        stop_line_y = int(stop_line_p * H / 100)
        lx, ly, lw, lh = light_rect

        # Smooth Light Detection
        raw_state = self.get_light_state(frame, light_rect)
        self.light_buffer.append(raw_state)
        stable_light = max(set(self.light_buffer), key=self.light_buffer.count)
        is_prohibited = stable_light in ["RED", "YELLOW"]

        # Vehicle Tracking
        results = self.model.track(frame, persist=True, conf=conf, verbose=False)
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                v_type = self.model.names[cls].capitalize()

                # Classification Counter
                if tid not in self.counted_ids:
                    self.counted_ids.add(tid)
                    self.stats["Total"] += 1
                    # Map custom classes to UI stats
                    label = v_type if v_type in self.stats else "Car"
                    self.stats[label] += 1

                # Violation Logic: Must cross the stop line + safety buffer
                if is_prohibited and y2 > (stop_line_y + safety_offset):
                    if tid not in self.violated_ids:
                        now = time.time()
                        if now - self.last_violation_time.get(tid, 0) > 3:
                            self.violated_ids.add(tid)
                            self.stats["Violation"] += 1
                            self.last_violation_time[tid] = now
                            
                            # Capture Union Bounding Box (Vehicle + Light)
                            ex1, ey1 = max(0, min(x1, lx) - 60), max(0, min(y1, ly) - 60)
                            ex2, ey2 = min(W, max(x2, lx+lw) + 60), min(H, max(y2, ly+lh) + 60)
                            ev_img = frame[ey1:ey2, ex1:ex2].copy()
                            
                            # Visual Proofing
                            l_color = COLOR_RED if stable_light == "RED" else COLOR_YELLOW
                            cv2.rectangle(ev_img, (lx-ex1, ly-ey1), (lx+lw-ex1, ly+lh-ey1), l_color, 4)
                            cv2.rectangle(ev_img, (x1-ex1, y1-ey1), (x2-ex1, y2-ey1), (0, 0, 255), 2)
                            
                            self.evidence.insert(0, {
                                "ID": tid, "Type": v_type, "img": ev_img, 
                                "Light": stable_light, "Time": datetime.now().strftime("%H:%M:%S")
                            })

                # Main Display Drawing
                box_color = COLOR_RED if tid in self.violated_ids else COLOR_GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{v_type} #{tid}", (x1, y1-10), 0, 0.6, box_color, 2)

        # Draw UI Overlays
        line_color = COLOR_GREEN
        if stable_light == "RED": line_color = COLOR_RED
        elif stable_light == "YELLOW": line_color = COLOR_YELLOW
        
        cv2.line(frame, (0, stop_line_y), (W, stop_line_y), line_color, 3)
        # Safety Buffer line (White)
        cv2.line(frame, (0, stop_line_y + safety_offset), (W, stop_line_y + safety_offset), (255, 255, 255), 1)
        cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), COLOR_YELLOW, 2)
        
        return frame, stable_light

# ==========================================================
# STREAMLIT INTERFACE
# ==========================================================
st.set_page_config(page_title="AI Traffic Monitor", layout="wide")

if 'core' not in st.session_state:
    st.session_state.core = TrafficCore()
core = st.session_state.core

with st.sidebar:
    st.header("⚙️ System Settings")
    video_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
    stop_p = st.slider("Stop Line Position (%)", 30, 95, 80)
    safety_off = st.slider("Violation Buffer (px)", 0, 150, 40)
    conf_val = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.45)
    
    st.subheader("🚥 Light ROI (%)")
    lx_p, ly_p = st.slider("ROI X", 0, 100, 85), st.slider("ROI Y", 0, 100, 10)
    lw_p, lh_p = st.slider("ROI Width", 1, 20, 8), st.slider("ROI Height", 1, 30, 15)

    if st.button("🗑️ Reset All Data"):
        core.reset()
        st.rerun()

st.title("🚦 Traffic Intelligence & Violation System")

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    col_vid, col_stat = st.columns([0.7, 0.3])
    with col_vid:
        v_placeholder = st.empty()
        st.subheader("📸 Violation Evidence")
        e_placeholder = st.empty()
        
    with col_stat:
        st.subheader("📊 Live Analytics")
        l_ui, v_ui, t_ui = st.empty(), st.empty(), st.empty()
        
        st.markdown("#### 🚗 Vehicle Classification")
        st_table = st.empty() # THIS IS THE COUNTING TABLE YOU ASKED FOR
        
        if core.evidence:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "x") as z:
                for i, ev in enumerate(core.evidence):
                    _, img_enc = cv2.imencode(".jpg", ev["img"])
                    z.writestr(f"violation_id_{ev['ID']}_{i}.jpg", img_enc.tobytes())
            st.download_button("📥 Download Evidence (ZIP)", buf.getvalue(), "violations.zip", "application/zip")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (1000, int(frame.shape[0] * 1000 / frame.shape[1])))
        H, W = frame.shape[:2]
        l_rect = [int(lx_p*W/100), int(ly_p*H/100), int(lw_p*W/100), int(lh_p*H/100)]
        
        out_f, light_s = core.process_frame(frame, 0, conf_val, stop_p, safety_off, l_rect)
        v_placeholder.image(out_f, channels="BGR", use_container_width=True)
        
        # UI Updates
        bg_color = "#22c55e" # Green
        if light_s == "RED": bg_color = "#ef4444"
        elif light_s == "YELLOW": bg_color = "#facc15"
        
        l_ui.markdown(f"<div style='background:{bg_color}; color:black; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>LIGHT STATE: {light_s}</div>", unsafe_allow_html=True)
        v_ui.metric("Violations Detected", core.stats["Violation"])
        t_ui.metric("Total Vehicle Flow", core.stats["Total"])
        
        # RENDER THE VEHICLE COUNTING TABLE
        df_stats = pd.DataFrame([
            {"Vehicle Type": k, "Count": v} 
            for k, v in core.stats.items() 
            if k not in ["Total", "Violation"]
        ])
        st_table.table(df_stats)

        with e_placeholder.container():
            for ev in core.evidence[:3]:
                st.image(ev["img"], caption=f"ID: {ev['ID']} | Type: {ev['Type']} | Light: {ev['Light']} | {ev['Time']}", channels="BGR")
    cap.release()
else:
    st.info("Upload a video file in the sidebar to begin traffic monitoring.")
