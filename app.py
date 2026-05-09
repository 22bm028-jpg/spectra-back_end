"""
Spectra Digitizer — Flask Backend API
======================================
Deploy on Render (render.com) with:
  pip install -r requirements.txt
  gunicorn app:app

Endpoints:
  POST /api/analyze          - Upload image, returns bounds + axis + curves
  POST /api/extract          - Upload image + axis params + masks → extracted CSV data
  POST /api/average          - Send CSV data → cluster averages
  GET  /health               - Health check
"""

import io, base64, re, warnings, csv
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Allow all origins (Netlify frontend)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COLOUR_TABLE = [
    ("Red",      "#e74c3c", ([0,  80, 80],  [12,255,255]), ([168,80,80],[180,255,255])),
    ("Orange",   "#e67e22", ([12, 80, 80],  [22,255,255]), None),
    ("Yellow",   "#f1c40f", ([22, 80,130],  [35,255,255]), None),
    ("Green",    "#27ae60", ([50, 40, 60],  [90,255,255]), None),
    ("Cyan",     "#16a085", ([85, 50, 50],  [108,255,255]),None),
    ("Blue",     "#2980b9", ([108,50, 40],  [128,255,255]),None),
    ("Violet",   "#8e44ad", ([128,20, 40],  [150,255,255]),None),
    ("Magenta",  "#c0392b", ([150,30, 40],  [168,255,255]),None),
    ("DarkGray", "#546e7a", ([0,  0,  40],  [180,25, 130]),None),
    ("Black",    "#212121", ([0,  0,   0],  [180,255,  55]),None),
]

PLOT_COLORS = [
    "#e74c3c","#2980b9","#27ae60","#8e44ad",
    "#e67e22","#16a085","#f1c40f","#c0392b","#546e7a","#212121",
]

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def decode_image(file_storage):
    """Decode an uploaded file into a CV2 BGR image."""
    data = file_storage.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def img_to_b64(img_bgr, quality=85):
    """Encode a CV2 BGR image as a base64 JPEG string."""
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')

def fig_to_b64(fig):
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def smooth_curve(y, window=13, poly=3):
    if len(y) < window + 2:
        return y.copy()
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(y) - (0 if len(y) % 2 == 1 else 1))
    w = w if w % 2 == 1 else w - 1
    if w < poly + 2:
        return y.copy()
    try:
        return savgol_filter(y, w, poly)
    except Exception:
        return y.copy()

def interpolate_to_grid(x, y, x_grid):
    order = np.argsort(x)
    x, y  = np.array(x)[order], np.array(y)[order]
    x, ui = np.unique(x, return_index=True)
    y     = y[ui]
    if len(x) < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(x_grid)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT BOUNDS DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_plot_bounds(img):
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kh    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w//8), 1))
    kv    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h//8)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
    vert  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)
    h_rows = np.where(horiz.sum(axis=1) > w * 0.25)[0]
    v_cols = np.where(vert.sum(axis=0)  > h * 0.25)[0]
    xl = int(v_cols[0])  if len(v_cols) >= 2 else int(w * 0.10)
    xr = int(v_cols[-1]) if len(v_cols) >= 2 else int(w * 0.95)
    yt = int(h_rows[0])  if len(h_rows) >= 2 else int(h * 0.05)
    yb = int(h_rows[-1]) if len(h_rows) >= 2 else int(h * 0.92)
    if xr - xl < 100: xl, xr = int(w * 0.10), int(w * 0.95)
    if yb - yt < 100: yt, yb = int(h * 0.05), int(h * 0.92)
    return {"x_left": xl, "x_right": xr, "y_top": yt, "y_bottom": yb}

# ─────────────────────────────────────────────────────────────────────────────
#  CURVE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _fg_mask(roi_bgr):
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:,:,2] >= 220) & (hsv[:,:,1] <= 30)
    vdark = hsv[:,:,2] < 15
    return (~(white | vdark)).astype(np.uint8) * 255

def _colour_mask(hsv, lo1, hi1, lo2=None, hi2=None, fg=None):
    m = cv2.inRange(hsv, np.array(lo1, np.uint8), np.array(hi1, np.uint8))
    if lo2 is not None:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, np.array(lo2, np.uint8), np.array(hi2, np.uint8)))
    if fg is not None:
        m = cv2.bitwise_and(m, fg)
    return m

def _clean_mask(mask):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    m  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)
    m  = cv2.morphologyEx(m,    cv2.MORPH_OPEN,  k3, iterations=1)
    m  = cv2.dilate(m, kv, iterations=1)
    return m

def _build_grid_mask(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, bwd = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    _, bwl = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//10), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h//10)))
    return cv2.bitwise_or(
        cv2.bitwise_or(cv2.morphologyEx(bwd, cv2.MORPH_OPEN, kh),
                       cv2.morphologyEx(bwd, cv2.MORPH_OPEN, kv)),
        cv2.bitwise_or(cv2.morphologyEx(bwl, cv2.MORPH_OPEN, kh),
                       cv2.morphologyEx(bwl, cv2.MORPH_OPEN, kv)))

def _trace_mask(mask, hsv_roi, x_off, y_off):
    h, w = mask.shape
    xs, ys = [], []
    for x in range(w):
        py = np.where(mask[:, x] > 0)[0]
        if len(py) == 0: continue
        wts = hsv_roi[py, x, 1].astype(float) + 1.0
        ys.append(float(np.average(py, weights=wts)))
        xs.append(float(x + x_off))
    if len(xs) < 20:
        return None
    xp = np.array(xs); yp = np.array(ys)
    x_full = np.arange(xp[0], xp[-1] + 1, dtype=float)
    return x_full, np.interp(x_full, xp, yp)

def _fallback_dark(roi_bgr, grid_mask):
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    h, w  = bw.shape
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w//8), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h//8)))
    ax = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh),
                        cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv))
    gm  = cv2.resize(grid_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cur = cv2.bitwise_and(bw, cv2.bitwise_not(cv2.bitwise_or(ax, gm)))
    return cv2.morphologyEx(cur, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

def extract_curves_from_image(img_clean, bounds, mode="multi", min_col_frac=0.06):
    h_img, w_img = img_clean.shape[:2]
    xl = max(0, bounds["x_left"] + 2);  xr = min(w_img, bounds["x_right"] - 2)
    yt = max(0, bounds["y_top"] + 2);   yb = min(h_img, bounds["y_bottom"] - 2)
    roi     = img_clean[yt:yb, xl:xr]
    roi_h, roi_w = roi.shape[:2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    fg      = _fg_mask(roi)
    grid    = _build_grid_mask(roi)
    min_cols = max(5, int(roi_w * min_col_frac))

    candidates = []
    for (name, hex_col, (lo1, hi1), extra) in COLOUR_TABLE:
        lo2, hi2 = (extra[0], extra[1]) if extra else (None, None)
        raw  = _colour_mask(hsv_roi, lo1, hi1, lo2, hi2, fg)
        mask = _clean_mask(raw)
        gm   = cv2.resize(grid, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        pure = cv2.bitwise_and(mask, cv2.bitwise_not(gm))
        dilp = cv2.dilate(pure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        sup  = cv2.bitwise_and(cv2.bitwise_and(mask, gm), cv2.bitwise_not(dilp))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(sup))
        n_px   = int(mask.astype(bool).sum())
        if n_px < 60: continue
        n_cols = int(np.array([(mask[:, c] > 0).any() for c in range(roi_w)]).sum())
        if n_cols < min_cols: continue
        candidates.append({"name": name, "hex": hex_col, "mask": mask, "n_px": n_px, "n_cols": n_cols})

    keep = [True] * len(candidates)
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            if not keep[i] or not keep[j]: continue
            mi = candidates[i]["mask"].astype(bool)
            mj = candidates[j]["mask"].astype(bool)
            inter   = int((mi & mj).sum())
            smaller = min(int(mi.sum()), int(mj.sum()))
            if smaller > 0 and inter / smaller > 0.55:
                drop = i if candidates[i]["n_px"] < candidates[j]["n_px"] else j
                keep[drop] = False

    results = []
    for cand, ok in zip(candidates, keep):
        if not ok: continue
        out = _trace_mask(cand["mask"], hsv_roi, xl, yt)
        if out is None: continue
        xp, yp = out
        results.append({"name": cand["name"], "hex": cand["hex"], "x_pixels": xp, "y_pixels": yp})

    if not results:
        fb  = _fallback_dark(roi, grid)
        out = _trace_mask(fb, hsv_roi, xl, yt)
        if out:
            xp, yp = out
            results.append({"name": "Curve", "hex": "#333333", "x_pixels": xp, "y_pixels": yp})

    if mode == "single" and len(results) > 1:
        results = [max(results, key=lambda r: len(r["x_pixels"]))]

    return results

def pixels_to_data(x_pix, y_pix, bounds, xmin, xmax, ymin, ymax):
    xl, xr = bounds["x_left"],  bounds["x_right"]
    yt, yb = bounds["y_top"],   bounds["y_bottom"]
    rw = max(xr - xl, 1); rh = max(yb - yt, 1)
    xd = xmin + (x_pix - xl) / rw * (xmax - xmin)
    yd = ymax - (y_pix - yt) / rh * (ymax - ymin)
    return xd, yd

# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTERING & AVERAGING
# ─────────────────────────────────────────────────────────────────────────────

def resample_curve(x, y, n=300):
    order = np.argsort(x); x, y = np.array(x)[order], np.array(y)[order]
    x, ui = np.unique(x, return_index=True); y = y[ui]
    if len(x) < 2: return np.full(n, np.nan)
    xg = np.linspace(x[0], x[-1], n)
    return interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)(xg)

def cluster_curves(curves_list, n_clusters=3, n_pts=300):
    if not curves_list: return [], []
    matrix = []
    for c in curves_list:
        rs = resample_curve(c["x_data"], c["y_data"], n_pts)
        rs = np.nan_to_num(rs)
        mn, mx = rs.min(), rs.max()
        if mx - mn > 1e-9: rs = (rs - mn) / (mx - mn)
        matrix.append(rs)
    matrix = np.array(matrix, dtype=float)

    try:
        from sklearn.cluster import KMeans
        k = min(n_clusters, len(curves_list))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix) if k >= 2 else np.zeros(len(curves_list), dtype=int)
    except ImportError:
        labels = np.zeros(len(curves_list), dtype=int)
        for i in range(1, len(curves_list)):
            best_cls, best_sim = 0, -1
            for cls in range(i):
                sim = float(np.dot(matrix[i], matrix[cls]) / (np.linalg.norm(matrix[i]) * np.linalg.norm(matrix[cls]) + 1e-9))
                if sim > best_sim: best_sim = sim; best_cls = cls
            labels[i] = best_cls if best_sim > 0.85 else labels.max() + 1

    x_all   = np.concatenate([c["x_data"] for c in curves_list])
    x_grid  = np.linspace(np.nanmin(x_all), np.nanmax(x_all), n_pts)
    cluster_infos = []
    for ci in range(int(labels.max()) + 1):
        members = [curves_list[i] for i in range(len(curves_list)) if labels[i] == ci]
        if not members: continue
        stack  = [interpolate_to_grid(m["x_data"], m["y_data"], x_grid) for m in members]
        arr    = np.array(stack, dtype=float)
        y_avg  = np.nanmean(arr, axis=0); y_avg = np.nan_to_num(y_avg)
        y_sm   = smooth_curve(y_avg, 25)
        cluster_infos.append({
            "cluster_id": ci,
            "n": len(members),
            "member_names": [m["name"] for m in members],
            "x_grid": x_grid.tolist(),
            "y_avg": y_avg.tolist(),
            "y_smooth": y_sm.tolist(),
        })
    return labels.tolist(), cluster_infos

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(curves, xlabel="Wavelength (nm)", ylabel="Absorbance", title="Extracted Spectra"):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('#fafafa')
    for i, c in enumerate(curves):
        col = c.get("hex", PLOT_COLORS[i % len(PLOT_COLORS)])
        ax.plot(c["x_data"], c["y_smooth"], color=col, lw=2, label=c["name"])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, color='#e0e0e0', lw=0.7, ls='--', alpha=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64

def draw_bounds_overlay(img_bgr, bounds):
    out = img_bgr.copy()
    xl, xr = bounds["x_left"], bounds["x_right"]
    yt, yb = bounds["y_top"],  bounds["y_bottom"]
    cv2.rectangle(out, (xl, yt), (xr, yb), (39, 174, 96), 2)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "Spectra Digitizer API"})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Step 1: Upload an image → get detected plot bounds + default axis values + annotated preview.
    Form data:
      image  (file)
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = decode_image(request.files['image'])
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    bounds     = auto_detect_plot_bounds(img)
    annotated  = draw_bounds_overlay(img, bounds)
    preview_b64 = img_to_b64(annotated)

    axis_defaults = {
        "xmin": 260.0, "xmax": 800.0,
        "ymin": 0.0,   "ymax": 1.0,
        "xlabel": "Wavelength (nm)",
        "ylabel": "Absorbance",
    }

    return jsonify({
        "bounds":       bounds,
        "axis":         axis_defaults,
        "preview_b64":  preview_b64,
        "img_width":    img.shape[1],
        "img_height":   img.shape[0],
    })


@app.route('/api/extract', methods=['POST'])
def extract():
    """
    Step 2: Upload image with axis values + optional masks → extracted curves + plot.
    JSON body (multipart):
      image   (file)
      xmin, xmax, ymin, ymax  (float, form fields)
      xlabel, ylabel           (str,   form fields)
      x_left, x_right, y_top, y_bottom  (int, form fields)
      mode    (str: 'single' | 'multi', default 'multi')
      smooth_window (int, default 13)
      masks   (JSON array of {x,y,w,h} rects in image pixels, optional)
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = decode_image(request.files['image'])
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Parse axis
    def fget(k, d): return float(request.form.get(k, d))
    def iget(k, d): return int(request.form.get(k, d))
    def sget(k, d): return request.form.get(k, d)

    xmin = fget('xmin', 260.0); xmax = fget('xmax', 800.0)
    ymin = fget('ymin', 0.0);   ymax = fget('ymax', 1.0)
    xlabel = sget('xlabel', 'Wavelength (nm)')
    ylabel = sget('ylabel', 'Absorbance')
    mode   = sget('mode', 'multi')
    smooth_window = iget('smooth_window', 13)

    bounds = {
        "x_left":   iget('x_left',   int(img.shape[1] * 0.10)),
        "x_right":  iget('x_right',  int(img.shape[1] * 0.95)),
        "y_top":    iget('y_top',     int(img.shape[0] * 0.05)),
        "y_bottom": iget('y_bottom',  int(img.shape[0] * 0.92)),
    }

    # Apply masks (whiten regions)
    import json as _json
    masks_raw = request.form.get('masks', '[]')
    try:
        masks = _json.loads(masks_raw)
    except Exception:
        masks = []

    img_clean = img.copy()
    for m in masks:
        x1, y1 = int(m.get('x', 0)), int(m.get('y', 0))
        x2, y2 = x1 + int(m.get('w', 0)), y1 + int(m.get('h', 0))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 > x1 and y2 > y1:
            img_clean[y1:y2, x1:x2] = 255

    raw_curves = extract_curves_from_image(img_clean, bounds, mode=mode)

    # Convert pixel → data coords; smooth
    output_curves = []
    for c in raw_curves:
        xd, yd = pixels_to_data(c['x_pixels'], c['y_pixels'], bounds, xmin, xmax, ymin, ymax)
        ys = smooth_curve(yd, smooth_window)
        # Subsample to max 800 pts for JSON efficiency
        idx = np.round(np.linspace(0, len(xd)-1, min(800, len(xd)))).astype(int)
        output_curves.append({
            "name":     c["name"],
            "hex":      c["hex"],
            "x_data":   xd[idx].tolist(),
            "y_data":   yd[idx].tolist(),
            "y_smooth": ys[idx].tolist(),
        })

    plot_b64 = plot_curves(output_curves, xlabel, ylabel) if output_curves else None

    # Build CSV string
    csv_io = io.StringIO()
    writer = csv.writer(csv_io)
    header = []
    for c in output_curves:
        header += [f"{c['name']}_X", f"{c['name']}_Y"]
    writer.writerow(header)

    max_len = max((len(c["x_data"]) for c in output_curves), default=0)
    for i in range(max_len):
        row = []
        for c in output_curves:
            if i < len(c["x_data"]):
                row += [f"{c['x_data'][i]:.6f}", f"{c['y_smooth'][i]:.6f}"]
            else:
                row += ["", ""]
        writer.writerow(row)

    return jsonify({
        "curves":   output_curves,
        "plot_b64": plot_b64,
        "csv":      csv_io.getvalue(),
        "xlabel":   xlabel,
        "ylabel":   ylabel,
        "n_curves": len(output_curves),
    })


@app.route('/api/average', methods=['POST'])
def average():
    """
    Step 3: Send multiple curves (from prior extractions) → cluster averages.
    JSON body:
      curves: [{name, x_data, y_data}, ...]
      n_clusters: int (default 3)
      xlabel, ylabel: str
    """
    body = request.get_json(force=True)
    if not body or 'curves' not in body:
        return jsonify({"error": "No curves provided"}), 400

    curves     = body['curves']
    n_clusters = int(body.get('n_clusters', 3))
    xlabel     = body.get('xlabel', 'Wavelength (nm)')
    ylabel     = body.get('ylabel', 'Absorbance')

    if len(curves) < 2:
        return jsonify({"error": "Need at least 2 curves for averaging"}), 400

    labels, cluster_infos = cluster_curves(curves, n_clusters)

    # Build plots for each cluster
    plots = []
    fig, axes = plt.subplots(1, len(cluster_infos),
                              figsize=(6 * len(cluster_infos), 5),
                              facecolor='white', squeeze=False)
    for idx, info in enumerate(cluster_infos):
        ax  = axes[0][idx]
        col = PLOT_COLORS[idx % len(PLOT_COLORS)]
        ax.set_facecolor('#fafafa')
        ax.plot(info["x_grid"], info["y_smooth"], color=col, lw=2.5,
                label=f"Avg (n={info['n']})")
        ax.fill_between(info["x_grid"], info["y_smooth"], alpha=0.12, color=col)
        ax.set_title(f"Cluster {info['cluster_id']+1}\n{', '.join(info['member_names'][:3])}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, color='#e0e0e0', lw=0.5, ls='--', alpha=0.7)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    fig.tight_layout()
    combined_b64 = fig_to_b64(fig)
    plt.close(fig)

    # CSV per cluster
    csv_out = {}
    for info in cluster_infos:
        cid = info["cluster_id"] + 1
        ci  = io.StringIO()
        w   = csv.writer(ci)
        w.writerow([f"Cluster_{cid}_X", f"Cluster_{cid}_Y_avg"])
        for x, y in zip(info["x_grid"], info["y_smooth"]):
            w.writerow([f"{x:.6f}", f"{y:.6f}"])
        csv_out[f"cluster_{cid}"] = ci.getvalue()

    return jsonify({
        "labels":        labels,
        "cluster_infos": cluster_infos,
        "plot_b64":      combined_b64,
        "csv_per_cluster": csv_out,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, port=5000)
