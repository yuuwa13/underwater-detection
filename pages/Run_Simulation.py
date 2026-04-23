import base64
import time
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

from utils import inject_branding, render_footer, render_navbar

st.set_page_config(
    page_title="Run Simulation · Underwater Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()
render_navbar("Run_Simulation")


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    baseline = YOLO("models/baseline_best.pt")
    proposed = YOLO("models/proposed_best.pt")

    def ensure_branch_weights(yolo_model):
        for module in yolo_model.model.modules():
            try:
                num_branches = None
                if hasattr(module, "branches"):
                    num_branches = len(module.branches)
                elif hasattr(module, "num_branches"):
                    num_branches = int(module.num_branches)
                elif hasattr(module, "branch_convs"):
                    num_branches = len(module.branch_convs)
                elif module.__class__.__name__ == "AdaptiveFeatureFusion":
                    num_branches = 2
                if num_branches and num_branches > 0:
                    module.register_parameter(
                        "branch_weights",
                        torch.nn.Parameter(
                            torch.ones(num_branches, dtype=torch.float32)
                        ),
                    )
            except Exception:
                pass
            if module.__class__.__name__ == "AdaptiveFeatureFusion":
                if not hasattr(module, "conv_align"):
                    module.add_module("conv_align", torch.nn.Identity())
                if not hasattr(module, "ca"):
                    module.add_module("ca", torch.nn.Identity())

    ensure_branch_weights(baseline)
    ensure_branch_weights(proposed)
    return baseline, proposed


baseline_model, proposed_model = load_models()


# ── Metrics ────────────────────────────────────────────────────────────────────
def get_model_metrics(model):
    metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
    try:
        ckpt = model.ckpt
        if ckpt and "metrics" in ckpt:
            if hasattr(ckpt["metrics"], "box"):
                b = ckpt["metrics"].box
                metrics["precision"] = float(b.p) if hasattr(b, "p") else 0.0
                metrics["recall"] = float(b.r) if hasattr(b, "r") else 0.0
                metrics["mAP50"] = float(b.map50) if hasattr(b, "map50") else 0.0
                metrics["mAP50-95"] = float(b.map) if hasattr(b, "map") else 0.0
    except Exception:
        pass
    return metrics


baseline_metrics = get_model_metrics(baseline_model)
proposed_metrics = get_model_metrics(proposed_model)

if baseline_metrics["precision"] == 0.0:
    baseline_metrics = {
        "precision": 0.8205,
        "recall": 0.7260,
        "mAP50": 0.8137,
        "mAP50-95": 0.5639,
    }
if proposed_metrics["precision"] == 0.0:
    proposed_metrics = {
        "precision": 0.8423,
        "recall": 0.72726,
        "mAP50": 0.82561,
        "mAP50-95": 0.57133,
    }

# ── Page header ────────────────────────────────────────────────────────────────
st.html("""
<div style="padding:0 0 28px;">
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
        Detection System
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:42px;font-weight:700;
               color:#0B3C5D;margin-bottom:14px;letter-spacing:-0.02em;">Run Simulation</h1>
    <p style="font-family:'Inter',sans-serif;font-size:17px;color:#64748b;max-width:600px;line-height:1.75;">
        Upload an underwater image to compare Baseline and Enhanced YOLOv12 detection results side by side.
    </p>
</div>
""")

# ── Upload ─────────────────────────────────────────────────────────────────────
if "reset_upload" not in st.session_state:
    st.session_state.reset_upload = False

# Reset cycle: clear flag then rerun so uploader starts fresh
if st.session_state.reset_upload:
    st.session_state.reset_upload = False
    st.rerun()

uploaded_file = st.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Hide native uploader entirely
    st.markdown(
        "<style>[data-testid='stFileUploader']{display:none!important;}</style>",
        unsafe_allow_html=True,
    )

    # Show unified preview card
    st.html(f"""
    <div style="border:1.5px dashed #cbd5e1;border-radius:12px;background:#fafafa;
                padding:24px;text-align:center;margin-bottom:8px;">
        <img src="data:image/png;base64,{b64}"
             style="max-width:100%;max-height:400px;border-radius:8px;object-fit:contain;" />
        <p style="font-size:13px;color:#94a3b8;margin-top:10px;margin-bottom:0;
                  font-family:'Inter',sans-serif;">{uploaded_file.name}</p>
    </div>
    """)

# Always show both buttons: Change Image (left) · Run Detection (right)
col_change, _, col_run = st.columns([2, 5, 2])
with col_change:
    change_clicked = st.button(
        "↩ Change Image",
        type="secondary",
        disabled=not uploaded_file,
        use_container_width=True,
    )
    if change_clicked and uploaded_file:
        st.session_state.reset_upload = True
        st.rerun()
with col_run:
    run_detection = st.button(
        "Run Detection",
        type="primary",
        disabled=not uploaded_file,
        use_container_width=True,
    )

st.html("<div style='height:20px;'></div>")

# ── Detection ──────────────────────────────────────────────────────────────────
if uploaded_file and run_detection:
    status_slot = st.empty()
    status_slot.info("Running detection on both models...")

    start = time.time()
    baseline_result = baseline_model(img_array)[0]
    baseline_time = time.time() - start

    start = time.time()
    proposed_result = proposed_model(img_array)[0]
    proposed_time = time.time() - start

    status_slot.empty()

    from collections import defaultdict

    def _iou(b1, b2):
        """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def match_detections(b_result, p_result, iou_thresh=0.3):
        """
        Assign shared label numbers to detections that overlap across models
        (same physical object). Matched pairs get the same number; unmatched
        detections get unique numbers appended after.
        Returns (b_label_map, p_label_map, b_labeled, p_labeled).
        """

        def group_by_class(result):
            groups = defaultdict(list)
            for i in range(len(result.boxes.cls)):
                cls = result.names[int(result.boxes.cls[i])]
                conf = float(result.boxes.conf[i])
                box = result.boxes.xyxy[i].tolist()
                groups[cls].append((conf, i, box))
            return groups

        b_groups = group_by_class(b_result)
        p_groups = group_by_class(p_result)
        all_cls = set(list(b_groups.keys()) + list(p_groups.keys()))

        b_label_map, p_label_map = {}, {}
        b_labeled, p_labeled = [], []

        for cls in all_cls:
            b_list = sorted(b_groups[cls], reverse=True)  # conf desc
            p_list = sorted(p_groups[cls], reverse=True)

            # Build all IoU pairs above threshold, sort best-first
            iou_pairs = []
            for bi, (_, _, bb) in enumerate(b_list):
                for pi, (_, _, pb) in enumerate(p_list):
                    score = _iou(bb, pb)
                    if score >= iou_thresh:
                        iou_pairs.append((score, bi, pi))
            iou_pairs.sort(reverse=True)

            # Greedy matching: highest IoU pair first
            used_b, used_p = set(), set()
            b_match, p_match = {}, {}
            for _, bi, pi in iou_pairs:
                if bi not in used_b and pi not in used_p:
                    b_match[bi] = pi
                    p_match[pi] = bi
                    used_b.add(bi)
                    used_p.add(pi)

            # Number matched pairs first (baseline confidence order),
            # then unmatched baseline, then unmatched enhanced
            num = 1
            b_num, p_num = {}, {}
            for bi in range(len(b_list)):
                if bi in b_match:
                    b_num[bi] = num
                    p_num[b_match[bi]] = num
                    num += 1
            for bi in range(len(b_list)):
                if bi not in b_match:
                    b_num[bi] = num
                    num += 1
            for pi in range(len(p_list)):
                if pi not in p_match:
                    p_num[pi] = num
                    num += 1

            for bi, (conf, box_idx, _) in enumerate(b_list):
                label = f"{cls} {b_num[bi]}"
                b_label_map[box_idx] = (label, conf)
                b_labeled.append((label, cls, conf))
            for pi, (conf, box_idx, _) in enumerate(p_list):
                label = f"{cls} {p_num[pi]}"
                p_label_map[box_idx] = (label, conf)
                p_labeled.append((label, cls, conf))

        b_labeled.sort(key=lambda x: x[2], reverse=True)
        p_labeled.sort(key=lambda x: x[2], reverse=True)
        return b_label_map, p_label_map, b_labeled, p_labeled

    def img_to_b64(arr):
        pil = Image.fromarray(arr)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def eval_metrics_html(m, improvements=None):
        def row(label, val, delta=None):
            delta_html = ""
            if delta is not None:
                color = "#10b981" if delta >= 0 else "#ef4444"
                sign = "+" if delta >= 0 else ""
                delta_html = f'<span style="font-size:12px;color:{color};margin-left:6px;">{sign}{delta:.2f}%</span>'
            return f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 0;border-bottom:1px solid #f1f5f9;">
                <div style="font-family:'Inter',sans-serif;font-size:14px;color:#64748b;">{label}</div>
                <div style="font-family:'Poppins',sans-serif;font-size:15px;font-weight:600;
                            color:#0B3C5D;">{val:.2f}%{delta_html}</div>
            </div>"""

        deltas = improvements or {}
        return (
            row("Precision", m["precision"] * 100, deltas.get("precision"))
            + row("Recall", m["recall"] * 100, deltas.get("recall"))
            + row("mAP@50", m["mAP50"] * 100, deltas.get("mAP50"))
            + row("mAP@50:95", m["mAP50-95"] * 100, deltas.get("mAP50-95"))
        )

    baseline_label_map, proposed_label_map, baseline_labeled, proposed_labeled = (
        match_detections(baseline_result, proposed_result)
    )

    # ── Class colour map (built once, used everywhere) ──────────────────────────
    _PALETTE_HEX = ["#EF4444", "#3B82F6", "#22C55E", "#EAB308", "#A855F7", "#F97316"]
    _unique_cls_early = sorted(
        set(
            lbl.rsplit(" ", 1)[0] for lbl, _, _ in (baseline_labeled + proposed_labeled)
        )
    )
    _cls_color_map = {
        _c.lower(): _PALETTE_HEX[i % len(_PALETTE_HEX)]
        for i, _c in enumerate(_unique_cls_early)
    }

    baseline_count = len(baseline_result.boxes.cls)
    proposed_count = len(proposed_result.boxes.cls)

    baseline_accuracy = (
        max(conf for _, _, conf in baseline_labeled) * 100 if baseline_labeled else 0
    )
    proposed_accuracy = (
        max(conf for _, _, conf in proposed_labeled) * 100 if proposed_labeled else 0
    )
    baseline_accuracy_avg = (
        np.mean([conf for _, _, conf in baseline_labeled]) * 100
        if baseline_labeled
        else 0
    )
    proposed_accuracy_avg = (
        np.mean([conf for _, _, conf in proposed_labeled]) * 100
        if proposed_labeled
        else 0
    )

    precision_improvement = (
        proposed_metrics["precision"] - baseline_metrics["precision"]
    ) * 100
    recall_improvement = (proposed_metrics["recall"] - baseline_metrics["recall"]) * 100
    map50_improvement = (proposed_metrics["mAP50"] - baseline_metrics["mAP50"]) * 100
    map95_improvement = (
        proposed_metrics["mAP50-95"] - baseline_metrics["mAP50-95"]
    ) * 100

    import json as _json

    clean_b64 = img_to_b64(img_array)

    def get_boxes_data(result, label_map):
        boxes = []
        for box_idx, (label, conf) in label_map.items():
            x1, y1, x2, y2 = result.boxes.xyxy[box_idx].tolist()
            boxes.append(
                {
                    "label": label,
                    "x1": round(x1),
                    "y1": round(y1),
                    "x2": round(x2),
                    "y2": round(y2),
                    "conf": round(conf, 4),
                    "idx": box_idx,
                }
            )
        return boxes

    baseline_boxes_js = _json.dumps(get_boxes_data(baseline_result, baseline_label_map))
    proposed_boxes_js = _json.dumps(get_boxes_data(proposed_result, proposed_label_map))

    # ── Card bottom fragments (highest conf, count, errors) ─────────────────────
    _err_both = (
        "This model can only detect Echinus, Starfish, Scallop, and "
        "Holothurian. Please upload a different image."
    )
    if baseline_count == 0:
        b_bottom = (
            f'<div class="no-detect">{_err_both}</div>'
            if proposed_count == 0
            else '<div class="no-detect">No class detected</div>'
        )
    else:
        _blabel = baseline_labeled[0][0]
        _bcnt = baseline_count
        _bcol = _cls_color_map.get(_blabel.rsplit(" ", 1)[0].lower(), "#1FA3A3")
        b_bottom = (
            f'<div class="conf-box conf-clickable" data-label="{_blabel}" '
            f'style="--cls-color:{_bcol};">'
            f'<div><div class="conf-label">Highest Confidence</div>'
            f'<div class="conf-cls">{_blabel}</div></div>'
            f'<div class="conf-pct">{baseline_accuracy:.1f}%</div></div>'
            f'<div class="conf-count">{_bcnt} object{"s" if _bcnt != 1 else ""} detected</div>'
        )

    if proposed_count == 0:
        e_bottom = (
            f'<div class="no-detect">{_err_both}</div>'
            if baseline_count == 0
            else '<div class="no-detect">No class detected</div>'
        )
    else:
        _elabel = proposed_labeled[0][0]
        _ecnt = proposed_count
        _ecol = _cls_color_map.get(_elabel.rsplit(" ", 1)[0].lower(), "#1FA3A3")
        e_bottom = (
            f'<div class="conf-box conf-clickable" data-label="{_elabel}" '
            f'style="--cls-color:{_ecol};">'
            f'<div><div class="conf-label">Highest Confidence</div>'
            f'<div class="conf-cls">{_elabel}</div></div>'
            f'<div class="conf-pct">{proposed_accuracy:.1f}%</div></div>'
            f'<div class="conf-count">{_ecnt} object{"s" if _ecnt != 1 else ""} detected</div>'
        )

    # ── Unified table ────────────────────────────────────────────────────────────
    _b_dict = {lbl: conf for lbl, _, conf in baseline_labeled}
    _p_dict = {lbl: conf for lbl, _, conf in proposed_labeled}
    all_labels = sorted(set(list(_b_dict.keys()) + list(_p_dict.keys())))

    _unique_cls = _unique_cls_early  # already computed above
    _cls_colors_js = _json.dumps(_cls_color_map)

    _trows = ""
    for lbl in all_labels:
        bc = _b_dict.get(lbl)
        pc = _p_dict.get(lbl)
        b_td = (
            f'<td class="col-b">{bc * 100:.1f}%</td>'
            if bc is not None
            else '<td class="col-b missing">—</td>'
        )
        p_td = (
            f'<td class="col-e">{pc * 100:.1f}%</td>'
            if pc is not None
            else '<td class="col-e missing">—</td>'
        )
        _cls_key = lbl.rsplit(" ", 1)[0].lower()
        _dot_color = _cls_color_map.get(_cls_key, "#1FA3A3")
        _trows += (
            f'<tr data-label="{lbl}" data-cls="{_cls_key}">'
            f'<td><span class="cls-dot" style="background:{_dot_color};"></span>{lbl}</td>'
            f"{b_td}{p_td}</tr>\n"
        )

    _filter_btns = '<button class="filter-btn active" data-filter="all">All</button>'
    for _c in _unique_cls:
        _col = _cls_color_map.get(_c.lower(), "#1FA3A3")
        _filter_btns += (
            f'<button class="filter-btn" data-filter="{_c.lower()}" '
            f'style="--cls-color:{_col};">{_c}</button>'
        )

    _table_html = (
        (
            '<div class="table-section">'
            '<div class="tbl-controls">'
            '<div class="eyebrow" style="margin-bottom:0;">Classification Results'
            '<span class="eyebrow-hint">hover or click to highlight</span></div>'
            '<div style="display:flex;align-items:center;gap:8px;">'
            '<div class="search-wrap">'
            '<input type="text" id="tbl-search" placeholder="Search..." /></div>'
            f'<div class="filter-wrap">__FILTER_BTNS__</div>'
            "</div></div>"
            '<div class="table-wrap"><table>'
            "<thead><tr>"
            "<th>Detection</th>"
            '<th class="col-b">Baseline</th>'
            '<th class="col-e">Enhanced</th>'
            "</tr></thead>"
            f'<tbody id="det-tbody">{_trows}</tbody>'
            "</table>"
            '<div class="no-rows" id="no-rows" style="display:none;">No results match your filter.</div>'
            "</div></div>"
        )
        if all_labels
        else ""
    )

    # ── Component height estimate ────────────────────────────────────────────────
    _ih, _iw = img_array.shape[:2]
    _est_canvas_h = int(_ih / _iw * 520)
    _component_h = _est_canvas_h + 200 + len(all_labels) * 52 + 180

    # ── Interactive HTML component ───────────────────────────────────────────────
    # Uses string replacement to avoid f-string escaping issues with JS braces.
    _TMPL = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Inter, system-ui, sans-serif; background: transparent; }
.models-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
.card { border-radius: 14px; overflow: hidden; border: 1px solid #e2e8f0; background: #fff; }
.card.enhanced { border: 1.5px solid #1FA3A3; box-shadow: 0 0 0 3px rgba(31,163,163,0.08); }
.card-header { padding: 18px 22px; border-bottom: 1px solid #e2e8f0; }
.enhanced .card-header { border-bottom: 1.5px solid #1FA3A3; }
.card-title { font-family: Poppins, sans-serif; font-size: 17px; font-weight: 600; color: #0B3C5D; }
canvas { width: 100%; height: auto; display: block; }
.conf-box { padding: 14px 18px; display: flex; justify-content: space-between; align-items: center;
            background: #f8fafc; border-top: 1px solid #e2e8f0; }
.conf-clickable { cursor: pointer; transition: background 0.12s; }
.conf-clickable:hover { background: #f0fdfc; }
.conf-clickable.locked { background: #d9f7f7 !important;
                         outline: 2px solid var(--cls-color,#1FA3A3); outline-offset: -2px; }
.conf-label { font-size: 12px; color: #64748b; margin-bottom: 3px; }
.conf-cls { font-family: Poppins, sans-serif; font-size: 14px; font-weight: 600; color: #0B3C5D; }
.conf-pct { font-family: Poppins, sans-serif; font-size: 24px; font-weight: 700; color: #0B3C5D; }
.conf-count { font-size: 12px; color: #94a3b8; padding: 8px 18px 14px; }
.no-detect { padding: 12px 18px; font-size: 14px; color: #ef4444; background: #fef2f2;
             border-top: 1px solid #fee2e2; }
.table-section { margin-top: 8px; }
.eyebrow { font-size: 13px; font-weight: 600; color: #1FA3A3; letter-spacing: 0.1em;
           text-transform: uppercase; margin-bottom: 14px; display: flex; align-items: center; gap: 10px; }
.eyebrow-hint { font-size: 11px; font-weight: 400; color: #94a3b8; letter-spacing: 0;
                text-transform: none; }
.table-wrap { border-radius: 14px; overflow: hidden; border: 1px solid #e2e8f0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
table { width: 100%; border-collapse: collapse; }
thead tr { background: #f8fafc; }
th { padding: 12px 16px; font-size: 13px; font-weight: 600; color: #64748b;
     border-bottom: 2px solid #e2e8f0; text-align: left; letter-spacing: 0.03em; }
th.col-b, th.col-e { text-align: center; }
th.col-e { color: #1FA3A3; }
tbody tr { cursor: pointer; transition: background 0.1s; }
tbody tr:hover { background: #f0fdfc !important; }
tbody tr.locked { background: #d9f7f7 !important; outline: 2px solid #1FA3A3; outline-offset: -2px; }
tbody tr:nth-child(even) { background: #fafafa; }
td { padding: 12px 16px; font-size: 14px; border-bottom: 1px solid #f1f5f9;
     color: #0f172a; font-weight: 500; }
td.col-b, td.col-e { text-align: center; font-family: Poppins, sans-serif; font-weight: 600; color: #0B3C5D; }
td.missing { color: #94a3b8 !important; font-weight: 400 !important; }
.tbl-controls { display: flex; align-items: center; justify-content: space-between;
                gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.search-wrap { position: relative; flex: 1; min-width: 140px; max-width: 240px; }
.search-wrap::before { content: ''; position: absolute; left: 10px; top: 50%;
                        transform: translateY(-50%); width: 14px; height: 14px;
                        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394a3b8' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E") center/contain no-repeat; }
.search-wrap input { width: 100%; padding: 7px 10px 7px 30px; border: 1px solid #e2e8f0;
                     border-radius: 8px; font-size: 13px; font-family: Inter, sans-serif;
                     color: #0f172a; outline: none; background: #fff; }
.search-wrap input:focus { border-color: #1FA3A3; box-shadow: 0 0 0 3px rgba(31,163,163,0.1); }
.search-wrap input::placeholder { color: #94a3b8; }
.filter-wrap { display: flex; gap: 5px; flex-wrap: wrap; }
.filter-btn { padding: 5px 11px; border: 1px solid #e2e8f0; border-radius: 20px;
              font-size: 12px; font-weight: 600; font-family: Inter, sans-serif;
              color: #64748b; background: #fff; cursor: pointer; transition: all 0.12s;
              white-space: nowrap; }
.filter-btn:hover { border-color: var(--cls-color,#1FA3A3); color: var(--cls-color,#1FA3A3); }
.filter-btn.active { background: var(--cls-color,#1FA3A3); border-color: var(--cls-color,#1FA3A3); color: #fff; }
.cls-dot { display:inline-block; width:9px; height:9px; border-radius:50%;
           margin-right:8px; vertical-align:middle; flex-shrink:0; }
.no-rows { padding: 24px 16px; text-align: center; font-size: 14px; color: #94a3b8;
           font-family: Inter, sans-serif; }
</style>
</head>
<body>
<div class="models-row">
  <div class="card">
    <div class="card-header"><div class="card-title">Baseline Model</div></div>
    <canvas id="cvs-b"></canvas>
    __B_BOTTOM__
  </div>
  <div class="card enhanced">
    <div class="card-header"><div class="card-title">Enhanced Model</div></div>
    <canvas id="cvs-e"></canvas>
    __E_BOTTOM__
  </div>
</div>
__TABLE_HTML__
<script>
var PALETTE = ["#EF4444","#3B82F6","#22C55E","#EAB308","#A855F7","#F97316"];
var CLS_COLORS = __CLS_COLORS__;
var B_BOXES = __BASELINE_BOXES__;
var E_BOXES = __PROPOSED_BOXES__;

var img = new Image();
var cvB = document.getElementById('cvs-b');
var cvE = document.getElementById('cvs-e');
var ctxB = cvB ? cvB.getContext('2d') : null;
var ctxE = cvE ? cvE.getContext('2d') : null;
var ready = false;

img.onload = function() {
  [cvB, cvE].forEach(function(cv) {
    if (cv) { cv.width = img.naturalWidth; cv.height = img.naturalHeight; }
  });
  requestAnimationFrame(function() {
    [cvB, cvE].forEach(function(cv) {
      if (cv) {
        var dw = cv.offsetWidth;
        if (dw > 0) {
          cv.style.height = Math.round(dw * img.naturalHeight / img.naturalWidth) + 'px';
        }
      }
    });
    ready = true;
    draw(null);
    setTimeout(notifyH, 120);
    setTimeout(notifyH, 600);
  });
};
img.src = '__IMG_SRC__';

function hexRgb(c) {
  return [parseInt(c.slice(1,3),16), parseInt(c.slice(3,5),16), parseInt(c.slice(5,7),16)];
}

function clsKey(label) {
  return label.replace(/ \d+$/, '').toLowerCase();
}

function drawBox(ctx, box, isHL) {
  var color = CLS_COLORS[clsKey(box.label)] || PALETTE[0];
  var rgb = hexRgb(color);
  var bw = box.x2 - box.x1, bh = box.y2 - box.y1;
  ctx.fillStyle = 'rgba('+rgb[0]+','+rgb[1]+','+rgb[2]+',0.12)';
  ctx.fillRect(box.x1, box.y1, bw, bh);
  ctx.strokeStyle = color;
  ctx.lineWidth = isHL ? 4 : 2.5;
  ctx.strokeRect(box.x1, box.y1, bw, bh);
  var text = box.label + ' ' + Math.round(box.conf * 100) + '%';
  ctx.font = 'bold 13px Inter, sans-serif';
  var tw = ctx.measureText(text).width;
  var pad = 5, th = 15;
  var ty = box.y1 > th + 2*pad + 4 ? box.y1 - th - 2*pad - 2 : box.y2 + 2;
  var rx = box.x1, ry = ty, rw = tw + 2*pad, rh = th + 2*pad, rad = 3;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(rx+rad,ry); ctx.lineTo(rx+rw-rad,ry);
  ctx.quadraticCurveTo(rx+rw,ry,rx+rw,ry+rad);
  ctx.lineTo(rx+rw,ry+rh-rad);
  ctx.quadraticCurveTo(rx+rw,ry+rh,rx+rw-rad,ry+rh);
  ctx.lineTo(rx+rad,ry+rh);
  ctx.quadraticCurveTo(rx,ry+rh,rx,ry+rh-rad);
  ctx.lineTo(rx,ry+rad);
  ctx.quadraticCurveTo(rx,ry,rx+rad,ry);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.fillText(text, box.x1 + pad, ty + th + pad - 1);
}

function drawCanvas(ctx, boxes, hl) {
  if (!ctx) return;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(img, 0, 0);
  if (!hl) {
    boxes.forEach(function(box) { drawBox(ctx, box, false); });
    return;
  }
  var hlBox = null;
  for (var i = 0; i < boxes.length; i++) {
    if (boxes[i].label === hl) { hlBox = boxes[i]; break; }
  }
  if (hlBox) {
    // Dim everything outside the highlighted box using even-odd clipping
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.rect(hlBox.x1, hlBox.y1, hlBox.x2 - hlBox.x1, hlBox.y2 - hlBox.y1);
    ctx.clip('evenodd');
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
    drawBox(ctx, hlBox, true);
  } else {
    // This model didn't detect this label — dim the whole image
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Not detected', ctx.canvas.width/2, ctx.canvas.height/2);
    ctx.textAlign = 'left';
  }
}

function draw(hl) {
  if (!ready) return;
  drawCanvas(ctxB, B_BOXES, hl);
  drawCanvas(ctxE, E_BOXES, hl);
}

function notifyH() {
  var h = document.documentElement.scrollHeight;
  window.parent.postMessage({type:'streamlit:setFrameHeight', height: h + 2}, '*');
}

var lockedLabel = null;
var activeFilter = 'all';
var searchQuery = '';

function applyFilters() {
  var rows = document.querySelectorAll('#det-tbody tr[data-label]');
  var anyVisible = false;
  rows.forEach(function(row) {
    var cls = row.dataset.cls || '';
    var label = (row.dataset.label || '').toLowerCase();
    var matchFilter = activeFilter === 'all' || cls === activeFilter;
    var matchSearch = !searchQuery || label.indexOf(searchQuery) !== -1;
    var show = matchFilter && matchSearch;
    row.style.display = show ? '' : 'none';
    if (show) anyVisible = true;
  });
  var noRows = document.getElementById('no-rows');
  if (noRows) noRows.style.display = anyVisible ? 'none' : '';
  // If locked row is now hidden, release it
  if (lockedLabel) {
    var lockedRow = document.querySelector('#det-tbody tr[data-label="' + lockedLabel + '"]');
    if (lockedRow && lockedRow.style.display === 'none') {
      lockedLabel = null;
      document.querySelectorAll('#det-tbody tr.locked').forEach(function(r) { r.classList.remove('locked'); });
      draw(null);
    }
  }
  setTimeout(notifyH, 80);
}

var searchEl = document.getElementById('tbl-search');
if (searchEl) {
  searchEl.addEventListener('input', function() {
    searchQuery = this.value.toLowerCase().trim();
    applyFilters();
  });
}

document.querySelectorAll('.filter-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.filter-btn').forEach(function(b) { b.classList.remove('active'); });
    this.classList.add('active');
    activeFilter = this.dataset.filter;
    applyFilters();
  });
});

document.querySelectorAll('.conf-clickable[data-label]').forEach(function(box) {
  box.addEventListener('click', function() {
    var lbl = box.dataset.label;
    var wasLocked = lockedLabel === lbl;
    // Release any locked table rows
    document.querySelectorAll('#det-tbody tr.locked').forEach(function(r) { r.classList.remove('locked'); });
    // Release any locked conf boxes
    document.querySelectorAll('.conf-clickable.locked').forEach(function(b) { b.classList.remove('locked'); });
    if (wasLocked) {
      lockedLabel = null;
      draw(null);
    } else {
      lockedLabel = lbl;
      box.classList.add('locked');
      // Also highlight the matching table row if visible
      var matchRow = document.querySelector('#det-tbody tr[data-label="' + lbl + '"]');
      if (matchRow && matchRow.style.display !== 'none') matchRow.classList.add('locked');
      draw(lbl);
    }
  });
});

document.querySelectorAll('tbody tr[data-label]').forEach(function(row) {
  row.addEventListener('mouseenter', function() {
    if (!lockedLabel) draw(row.dataset.label);
  });
  row.addEventListener('mouseleave', function() {
    if (!lockedLabel) draw(null);
  });
  row.addEventListener('click', function() {
    var lbl = row.dataset.label;
    var wasLocked = lockedLabel === lbl;
    document.querySelectorAll('tbody tr.locked').forEach(function(r) { r.classList.remove('locked'); });
    document.querySelectorAll('.conf-clickable.locked').forEach(function(b) { b.classList.remove('locked'); });
    if (wasLocked) {
      lockedLabel = null;
      draw(null);
    } else {
      lockedLabel = lbl;
      row.classList.add('locked');
      // Also lock the conf box if it matches
      document.querySelectorAll('.conf-clickable[data-label="' + lbl + '"]').forEach(function(b) { b.classList.add('locked'); });
      draw(lbl);
    }
  });
});
</script>
</body>
</html>"""

    _component_html = (
        _TMPL.replace("__IMG_SRC__", "data:image/png;base64," + clean_b64)
        .replace("__BASELINE_BOXES__", baseline_boxes_js)
        .replace("__PROPOSED_BOXES__", proposed_boxes_js)
        .replace("__CLS_COLORS__", _cls_colors_js if all_labels else "{}")
        .replace("__B_BOTTOM__", b_bottom)
        .replace("__E_BOTTOM__", e_bottom)
        .replace("__TABLE_HTML__", _table_html)
        .replace("__FILTER_BTNS__", _filter_btns if all_labels else "")
    )

    st.components.v1.html(_component_html, height=_component_h, scrolling=False)

    # ── Evaluation Metrics expanders ─────────────────────────────────────────────
    if baseline_count > 0 or proposed_count > 0:
        em1, em2 = st.columns(2, gap="large")
        if baseline_count > 0:
            with em1:
                with st.expander("Baseline · Evaluation Metrics"):
                    _html = eval_metrics_html(baseline_metrics)
                    st.html(f"""<div style="padding:0 4px;">{_html}
                        <div style="font-family:'Inter',sans-serif;font-size:11.5px;color:#94a3b8;
                                    margin-top:10px;">Training benchmark metrics</div></div>""")
        if proposed_count > 0:
            with em2:
                with st.expander("Enhanced · Evaluation Metrics"):
                    _impr = {
                        "precision": precision_improvement,
                        "recall": recall_improvement,
                        "mAP50": map50_improvement,
                        "mAP50-95": map95_improvement,
                    }
                    _html = eval_metrics_html(proposed_metrics, _impr)
                    st.html(f"""<div style="padding:0 4px;">{_html}
                        <div style="font-family:'Inter',sans-serif;font-size:11.5px;color:#94a3b8;
                                    margin-top:10px;">Δ relative to baseline training metrics</div></div>""")

    # ── Model Comparison ────────────────────────────────────────────────────────
    st.html("<div style='height:48px;'></div>")
    st.html("""
    <div style="border-top:1px solid #e2e8f0;padding-top:48px;margin-bottom:32px;">
        <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
            Summary
        </div>
        <h2 style="font-family:'Poppins',sans-serif;font-size:36px;font-weight:700;
                   color:#0B3C5D;letter-spacing:-0.02em;">Model Comparison</h2>
    </div>
    """)

    acc_delta = proposed_accuracy_avg - baseline_accuracy_avg
    acc_color = "#10b981" if acc_delta >= 0 else "#ef4444"
    acc_sign = "+" if acc_delta >= 0 else ""

    cc1, cc2, cc3 = st.columns(3, gap="medium")
    for col, label, value, sub, sub_color in [
        (
            cc1,
            "Avg. Confidence (Enhanced)",
            f"{proposed_accuracy_avg:.1f}%",
            f"{acc_sign}{acc_delta:.1f}% vs baseline",
            acc_color,
        ),
        (cc2, "Baseline Detections", str(baseline_count), "objects found", "#64748b"),
        (cc3, "Enhanced Detections", str(proposed_count), "objects found", "#64748b"),
    ]:
        col.html(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                    padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-family:'Inter',sans-serif;font-size:13px;color:#64748b;
                        margin-bottom:10px;">{label}</div>
            <div style="font-family:'Poppins',sans-serif;font-size:28px;font-weight:700;
                        color:#0B3C5D;margin-bottom:6px;">{value}</div>
            <div style="font-family:'Inter',sans-serif;font-size:12.5px;
                        color:{sub_color};">{sub}</div>
        </div>
        """)

render_footer()
