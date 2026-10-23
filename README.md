# Minimal Local VR Video Player — Technical Architecture & Extensibility Guide

A lightweight, high-performance, dependency-free 360°/180° VR video player designed for local directory streaming. It uses native WebGL for GPU-accelerated rendering, custom matrix math for camera controls, and a Python backend supporting HTTP Range requests.

---

## 🛠️ Technology Stack

### 1. Frontend Core & Rendering

- **HTML5 `<video>` API:** Handles media decoding, playback state, and streaming buffers natively without heavy media frameworks.
- **WebGL (Web Graphics Library):** Provides low-level, hardware-accelerated GPU rendering by projecting video textures onto various geometry meshes:
  - **360° Sphere (`createSphere`):** Full spherical wrap for full 360° media.
  - **180° Dome (`createDome`):** Partial hemispherical projection for 180° cinematic VR.
  - **Fisheye Plane (`createPlane`):** Flat rectangular mapping optimized for fisheye-lensed footage.
- **Custom GLSL Shaders:**
  - *Vertex Shader:* Manages 3D transformations via a Model-View-Projection (MVP) matrix and handles UV coordinate mapping.
  - *Fragment Shader:* Samples the video texture and handles dynamic UV cropping via a stereoscopic format uniform (`uFormat`) to support split-screen 3D layouts.

### 2. Interaction, Navigation & Gesture Control

- **Mouse & Touch Events (`mousedown`, `mousemove`, `touchstart`, `touchmove`):** Tracks pointer/swipe deltas to calculate camera pitch (rotation X) and yaw (rotation Y).
- **Pinch-to-Zoom & Wheel Events:** Implements dynamic Field of View (FOV) scaling (`cameraZoom`) and gesture-locked horizontal trackpad seeking or zooming.
- **Keyboard Shortcuts:** Provides complete desktop control over playback, seeking, volume, muting, and fullscreen mode without relying solely on UI elements.
- **Fullscreen API:** Supports standard cross-platform browser full-screen requests with mobile fallbacks (`webkitEnterFullScreen`).

### 3. Backend & File Streaming

- **Python 3 (`http.server` & `socketserver`):** A custom multi-threaded Python server (`serve.py`) with address reuse enabled to prevent port-binding conflicts.
- **HTTP Range Requests (`206 Partial Content`):** Crucial for media streaming, allowing browsers and mobile devices to seek forward/backward instantly without downloading the entire video file.
- **Directory Auto-Discovery API (`/api/files`):** Automatically scans the server's working directory on load, filtering for valid video formats (`.mp4`, `.webm`, `.mkv`, etc.) and returning a JSON playlist.

---

## 📂 Project Structure

```text
├── serve.py        # Python backend (HTTP Range server + directory scanner API)
└── index.html      # Standalone frontend (WebGL engine, UI overlays, gesture controls)

```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `Space` / `K` | Play / Pause |
| `F` | Toggle Fullscreen |
| `M` | Toggle Mute |
| `Left Arrow` | Skip backward 5 seconds |
| `Right Arrow` | Skip forward 5 seconds |
| `Up Arrow` | Increase volume (+10%) |
| `Down Arrow` | Decrease volume (-10%) |

---

## 🚀 How to Extend & Modify

### 1. Adding Support for New Video Formats

To support additional video codecs or file types, update the `VIDEO_EXTENSIONS` tuple in `serve.py`:

```python
VIDEO_EXTENSIONS = ('.mp4', '.m4v', '.webm', '.mov', '.mkv', '.avi', '.flv')

```

### 2. Extending Stereoscopic 3D Layouts

The fragment shader includes built-in UV coordinate splitting controlled by the `uFormat` uniform integer:

* `0`: Monoscopic (2D/Flat)
* `1` / `2`: Side-by-Side (Left Eye / Right Eye)
* `3` / `4`: Top-Bottom (Top / Bottom)

To add a custom layout, update the `if-else` block inside the fragment shader (`fsSource`) to calculate custom UV multipliers and offsets.

### 3. Adding WebXR / Headset Support (Meta Quest / Vision Pro)

Because this player uses a raw WebGL canvas (`#canvas`), you can integrate the **WebXR Device API** directly:

* Request an `immersive-vr` session using `navigator.xr.requestSession('immersive-vr')`.
* Bind the WebGL context to the WebXR session layer (`session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) })`).
* Replace the mouse/touch rotation matrix with device orientation quaternions provided by `xrSession.requestAnimationFrame`.