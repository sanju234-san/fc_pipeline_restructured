"""Browser-based interactive EEG image viewer rendered as a Chainlit HTML element.

This generates a self-contained HTML page with:
- Mouse-wheel / pinch zoom
- Click-drag pan
- Slider for horizontal sliding
- Zoom in / out / fit / reset buttons

The HTML is served as a ``cl.Html`` element inside the Chainlit sidebar so
the user never leaves the browser.  No desktop window, no Qt required.
"""

from __future__ import annotations

import base64
from pathlib import Path


def _image_to_base64(path: str | Path) -> str:
    """Read an image file and return a base64-encoded data URI."""
    path = Path(path)
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix, "image/png")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_plotly_figure(image_path: str | Path, title: str = "EEG Plot"):
    """Build an interactive Plotly figure from an image file for Chainlit.

    Features:
    - Mouse scroll-wheel zoom in/out
    - Click and drag to pan/slide across the image
    - Double-click to reset view
    - Modebar with zoom box, pan, zoom in, zoom out, autoscale, reset
    - Works directly with ``cl.Plotly(figure=fig, display='side', size='large')``
    """
    from PIL import Image
    import plotly.graph_objects as go

    path = Path(image_path)
    img = Image.open(path)
    w, h = img.size

    uri = _image_to_base64(path)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, w],
            y=[0, h],
            mode="markers",
            marker=dict(opacity=0),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_layout_image(
        dict(
            source=uri,
            xref="x",
            yref="y",
            x=0,
            y=h,
            sizex=w,
            sizey=h,
            sizing="stretch",
            layer="below",
        )
    )
    fig.update_xaxes(
        range=[0, w],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[0, h],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title=dict(
            text=f"🧠 {title}",
            font=dict(color="#e94560", size=14, family="Segoe UI, sans-serif"),
        ),
        margin=dict(l=5, r=5, t=35, b=5),
        paper_bgcolor="#16213e",
        plot_bgcolor="#0f0f1a",
        dragmode="pan",
        autosize=True,
    )
    return fig


def build_viewer_html(image_path: str | Path, title: str = "EEG Plot") -> str:
    """Return a self-contained HTML string with zoom/pan/slide controls.

    The image is embedded as a base64 data URI so the HTML works without
    a file server and can be sent directly via ``cl.Html``.

    Parameters
    ----------
    image_path : str | Path
        Absolute path to the PNG/JPEG plot.
    title : str
        Label shown in the viewer header.

    Returns
    -------
    str
        Full HTML document (no external dependencies).
    """
    data_uri = _image_to_base64(image_path)
    safe_title = str(title).replace("<", "&lt;").replace(">", "&gt;")

    # Inline HTML + CSS + JS — zero external dependencies
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0f0f1a;font-family:'Segoe UI',Arial,sans-serif;color:#eaeaea;user-select:none;overflow:hidden;}}
  #header{{
    background:#16213e;border-bottom:2px solid #e94560;padding:6px 12px;
    display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  }}
  #header h3{{font-size:13px;color:#e94560;font-weight:700;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .btn{{
    background:#1a1a2e;color:#eaeaea;border:1px solid #e94560;border-radius:5px;
    padding:4px 10px;font-size:12px;cursor:pointer;transition:background .15s,color .15s;
    white-space:nowrap;
  }}
  .btn:hover{{background:#e94560;color:#fff;}}
  #zoom-label{{font-size:12px;color:#e94560;font-weight:bold;min-width:42px;text-align:right;}}
  #slide-row{{
    background:#0f0f1a;padding:4px 10px;display:flex;align-items:center;gap:8px;
    border-bottom:1px solid #222;
  }}
  #slide-row label{{font-size:11px;color:#aaa;white-space:nowrap;}}
  #hslider{{flex:1;accent-color:#e94560;cursor:pointer;}}
  #canvas-wrap{{
    position:relative;overflow:hidden;
    width:100%;height:calc(100vh - 76px);
    cursor:grab;background:#0f0f1a;
  }}
  #canvas-wrap.dragging{{cursor:grabbing;}}
  #img{{
    position:absolute;transform-origin:0 0;
    image-rendering:auto;
    transition:none;
  }}
  #hint{{position:absolute;bottom:6px;right:10px;font-size:10px;color:#555;pointer-events:none;}}
</style>
</head>
<body>
<div id="header">
  <h3 title="{safe_title}">🧠 {safe_title}</h3>
  <button class="btn" onclick="zoomIn()">Zoom +</button>
  <button class="btn" onclick="zoomOut()">Zoom −</button>
  <button class="btn" onclick="zoomFit()">Fit</button>
  <button class="btn" onclick="zoom100()">1:1</button>
  <button class="btn" onclick="resetView()">Reset</button>
  <span id="zoom-label">100%</span>
</div>
<div id="slide-row">
  <label>◀ Slide ▶</label>
  <input id="hslider" type="range" min="0" max="1000" value="500" oninput="onSlide(this.value)"/>
  <label id="slide-pct">50%</label>
</div>
<div id="canvas-wrap" id="wrap">
  <img id="img" src="{data_uri}" draggable="false" onload="init()"/>
  <div id="hint">Scroll = zoom · Drag = pan · Slider = slide</div>
</div>

<script>
const wrap = document.getElementById('canvas-wrap');
const img  = document.getElementById('img');
const zoomLbl = document.getElementById('zoom-label');
const slidePct = document.getElementById('slide-pct');
const hslider  = document.getElementById('hslider');

let scale = 1, ox = 0, oy = 0;
let dragging = false, lastX = 0, lastY = 0;
const ZOOM_STEP = 1.2, MIN_Z = 0.05, MAX_Z = 20;

function clampOx(){{
  const maxOx = 0;
  const minOx = wrap.clientWidth  - img.naturalWidth  * scale;
  ox = Math.min(maxOx, Math.max(minOx, ox));
}}
function clampOy(){{
  const maxOy = 0;
  const minOy = wrap.clientHeight - img.naturalHeight * scale;
  oy = Math.min(maxOy, Math.max(minOy, oy));
}}
function apply(){{
  img.style.transform = `translate(${{ox}}px,${{oy}}px) scale(${{scale}})`;
  zoomLbl.textContent = Math.round(scale*100) + '%';
  // sync horizontal slider
  const maxOx = 0;
  const minOx = wrap.clientWidth  - img.naturalWidth  * scale;
  const range = maxOx - minOx;
  if(range > 0){{
    const pct = (ox - minOx) / range;
    hslider.value = Math.round(pct * 1000);
    slidePct.textContent = Math.round(pct * 100) + '%';
  }}
}}
function init(){{
  zoomFit();
}}
function zoomFit(){{
  const sw = wrap.clientWidth / img.naturalWidth;
  const sh = (wrap.clientHeight) / img.naturalHeight;
  scale = Math.min(sw, sh);
  ox = (wrap.clientWidth  - img.naturalWidth  * scale) / 2;
  oy = (wrap.clientHeight - img.naturalHeight * scale) / 2;
  apply();
}}
function zoom100(){{
  const cx = wrap.clientWidth/2, cy = wrap.clientHeight/2;
  ox = cx - img.naturalWidth/2;
  oy = cy - img.naturalHeight/2;
  scale = 1;
  apply();
}}
function zoomAt(factor, cx, cy){{
  const newScale = Math.min(MAX_Z, Math.max(MIN_Z, scale * factor));
  if(newScale === scale) return;
  ox = cx - (cx - ox) * (newScale / scale);
  oy = cy - (cy - oy) * (newScale / scale);
  scale = newScale;
  clampOy();
  apply();
}}
function zoomIn(){{ zoomAt(ZOOM_STEP, wrap.clientWidth/2, wrap.clientHeight/2); }}
function zoomOut(){{ zoomAt(1/ZOOM_STEP, wrap.clientWidth/2, wrap.clientHeight/2); }}
function resetView(){{ zoomFit(); }}
function onSlide(val){{
  const minOx = wrap.clientWidth  - img.naturalWidth  * scale;
  const maxOx = 0;
  const range = maxOx - minOx;
  if(range > 0){{
    ox = minOx + (val / 1000) * range;
    apply();
  }}
  slidePct.textContent = Math.round(val/10) + '%';
}}

// Wheel zoom
wrap.addEventListener('wheel', e => {{
  e.preventDefault();
  const rect = wrap.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  zoomAt(e.deltaY < 0 ? ZOOM_STEP : 1/ZOOM_STEP, cx, cy);
}}, {{passive: false}});

// Drag pan
wrap.addEventListener('mousedown', e => {{
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  wrap.classList.add('dragging');
}});
window.addEventListener('mousemove', e => {{
  if(!dragging) return;
  ox += e.clientX - lastX;
  oy += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  clampOx(); clampOy(); apply();
}});
window.addEventListener('mouseup', () => {{
  dragging = false; wrap.classList.remove('dragging');
}});

// Touch support
let lastPinchDist = null;
wrap.addEventListener('touchstart', e => {{
  if(e.touches.length === 1){{
    dragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  }} else if(e.touches.length === 2){{
    lastPinchDist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
  }}
}}, {{passive: true}});
wrap.addEventListener('touchmove', e => {{
  if(e.touches.length === 1 && dragging){{
    ox += e.touches[0].clientX - lastX;
    oy += e.touches[0].clientY - lastY;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    clampOx(); clampOy(); apply();
  }} else if(e.touches.length === 2 && lastPinchDist !== null){{
    const dist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
    const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
    const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    const rect = wrap.getBoundingClientRect();
    zoomAt(dist / lastPinchDist, cx - rect.left, cy - rect.top);
    lastPinchDist = dist;
  }}
}}, {{passive: true}});
wrap.addEventListener('touchend', () => {{
  dragging = false; lastPinchDist = null;
}});

// Keyboard
window.addEventListener('keydown', e => {{
  if(e.key === '+' || e.key === '=') zoomIn();
  else if(e.key === '-') zoomOut();
  else if(e.key === 'f' || e.key === 'F') zoomFit();
  else if(e.key === '0') zoom100();
  else if(e.key === 'ArrowLeft')  {{ ox -= 40; clampOx(); apply(); }}
  else if(e.key === 'ArrowRight') {{ ox += 40; clampOx(); apply(); }}
  else if(e.key === 'ArrowUp')    {{ oy -= 40; clampOy(); apply(); }}
  else if(e.key === 'ArrowDown')  {{ oy += 40; clampOy(); apply(); }}
}});
</script>
</body>
</html>"""
    return html
