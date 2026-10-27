# AGENTS.md — minislam: a minimal monocular ORB-SLAM in Python

## Purpose

Build a small, readable monocular SLAM system that runs on a laptop webcam (or any USB camera) on Windows. It follows the structure of ORB-SLAM3 (ORB features, map points, keyframes, covisibility, local bundle adjustment) but implements only the core. A GUI shows the live video with the landmarks in use, plus a 3D view of the map and the camera trajectory.

This is a **learning implementation**. The owner will later rewrite it by hand, so clarity beats performance and cleverness throughout. Every module should be understandable on its own and should map clearly onto the equivalent part of ORB-SLAM3.

## Working rules for the agent

- **Ask before you guess.** If something in this spec is ambiguous, contradictory, or turns out infeasible, stop and ask the owner. Do not silently change scope, swap libraries, or drop features.
- **Ask before adding any dependency** not listed under "Environment."
- **Build milestone by milestone** (see "Milestones"). Each milestone must run end to end and meet its acceptance criteria before you start the next. Commit at the end of each milestone with a message naming it.
- **Write tests as you go** (see "Testing"). Run `pytest` before declaring a milestone done.
- **Explain the code for a human learner:**
  - Each module starts with a docstring explaining what it does, the math involved, and which ORB-SLAM3 source file it corresponds to (e.g. `Tracking.cc`, `LocalMapping.cc`, `Optimizer.cc`, `ORBmatcher.cc`).
  - Mark every high-level OpenCV call that hides real algorithmic work with a `# BLACKBOX:` comment, saying what it computes and what a by-hand replacement would involve. This applies to `findEssentialMat`, `recoverPose`, `triangulatePoints`, `solvePnPRansac`, and `calibrateCamera`.
- **Keep parameters in config.** No magic numbers buried in code: tunable values live in `config/default.yaml` with a comment explaining each one.
- **Use type hints and small functions.** Avoid metaprogramming and deep class hierarchies.

## Non-goals (v1)

These are out of scope for v1. Do not implement them unless the owner asks:

- IMU fusion
- Stereo, RGB-D, or multi-camera
- Loop closure, DBoW vocabularies, or Sim(3) pose graph optimization
- Map save/load and multi-map (Atlas)
- Keyframe culling
- A separate local-mapping thread
- GPU acceleration
- Real-time guarantees

## Environment

- **OS:** Windows 10/11. Use `pathlib` for paths and nothing POSIX-only.
- **Python:** 3.13 (changed from 3.11 with the owner's approval: 3.13 is what is installed, and every dependency ships 3.13 wheels). Ask the owner before changing the version again.
- **Virtual environment:** `py -3.13 -m venv .venv`, then `.venv\Scripts\activate`, then `pip install -r requirements.txt`.

### Dependencies (`requirements.txt`)

| Package | Use |
|---|---|
| `numpy` | All linear algebra |
| `scipy` | `scipy.optimize.least_squares` (sparse Jacobian) for pose-only optimization and bundle adjustment; `scipy.spatial.transform.Rotation` |
| `opencv-python` | Camera capture, ORB, matching, geometric solvers, calibration |
| `PySide6` | GUI window and event loop |
| `pyqtgraph` + `PyOpenGL` | 3D map view (`pyqtgraph.opengl.GLViewWidget`) |
| `PyYAML` | Config files |
| `pytest` | Tests |
| `evo` (optional, dev only) | Trajectory evaluation against ground truth |

## Repository layout

```
minislam/
  AGENTS.md
  README.md                # how to install, calibrate, run, evaluate
  requirements.txt
  run.py                   # entry point: python run.py --source 0 --camera config/camera.yaml
  config/
    default.yaml           # all SLAM + GUI parameters
    camera.yaml            # produced by tools/calibrate.py
  minislam/
    __init__.py
    config.py              # load/validate YAML into dataclasses
    camera.py              # PinholeCamera: K, distortion, project, undistort_points
    source.py              # FrameSource: webcam / video file / image folder (+ timestamps)
    features.py            # ORB extraction with grid distribution, per-octave sigma
    frame.py               # Frame: keypoints, descriptors, pose estimate, map-point associations
    map.py                 # MapPoint, KeyFrame, Map (incl. covisibility graph)
    matcher.py             # descriptor matching: by projection, brute force + ratio, epipolar-guided
    geometry.py            # SE(3) helpers, triangulation checks, parallax, reprojection error
    initializer.py         # two-view monocular map initialization
    tracking.py            # per-frame tracking state machine
    local_mapping.py       # keyframe processing: triangulation, point culling, local BA call
    optimizer.py           # pose-only optimization and local/full bundle adjustment
    system.py              # SlamSystem: owns everything; process(frame) -> Snapshot
    snapshot.py            # immutable data passed to the GUI
    io.py                  # save trajectory (TUM format), export map points (PLY)
    gui/
      app.py               # QMainWindow, worker thread, keyboard shortcuts
      overlay.py           # draws landmarks/HUD onto the video frame
      map_view.py          # 3D points, keyframe frustums, trajectory
  tools/
    calibrate.py           # checkerboard calibration -> config/camera.yaml
    record.py              # record webcam to video + timestamps for repeatable runs
  tests/
    test_geometry.py
    test_initializer.py
    test_tracking_pose.py
    test_bundle_adjustment.py
    synthetic.py           # synthetic scene/trajectory generator used by tests
```

## Conventions

- **Camera frame:** OpenCV convention. x right, y down, z forward.
- **World frame:** the camera frame of the first initialization keyframe.
- **Poses:** stored as `T_cw` (4×4, world → camera), the same as ORB-SLAM3. Camera center is `C = -R_cwᵀ t_cw`. Name variables with explicit frames (`T_cw`, `T_wc`, `p_w`, `p_c`).
- **Rotation parameters in optimization:** axis-angle (Rodrigues) vectors.
- **Distortion:** undistort keypoint *coordinates* (`cv2.undistortPoints` with `P=K`), not whole images. All geometry works on undistorted pixel coordinates.
- **Monocular scale:** arbitrary. After initialization, normalize so the median scene depth equals 1. Document this clearly in the README.
- **Measurement uncertainty:** a keypoint at pyramid octave `o` has `σ² = scaleFactor^(2o)`. Weight reprojection residuals by `1/σ`. For outlier tests use the χ² threshold 5.991 (2 DOF, 95%) on `‖r‖²/σ²`.

## Core data structures

**`Frame`**
- id, timestamp, image (grayscale, for the GUI only)
- keypoints (raw and undistorted), descriptors (N×32 uint8), octaves
- `T_cw` (estimate or None)
- `map_points: list[MapPoint | None]` (length N)
- `outlier: np.ndarray[bool]`

**`KeyFrame`**
- Everything a `Frame` has, made permanent once inserted
- covisibility weights `dict[KeyFrame, int]` (number of shared map points)

**`MapPoint`**
- id, `p_w` (3,)
- observations `dict[KeyFrame, int]` (keyframe → keypoint index)
- representative descriptor: the observed descriptor with the smallest median Hamming distance to the others
- mean viewing direction
- counters `n_visible` / `n_found`, used for culling
- `is_bad` flag

**`Map`**
- keyframes, map points
- `add_*` / `remove_*` methods that keep observations and covisibility consistent in both directions

**`Snapshot`**
- deep-copied, GUI-safe data:
  - current image
  - keypoint positions with a status per keypoint (tracked inlier / outlier / unmatched)
  - tracking state, timing, counts
  - all map point positions, plus which are in the current local map
  - keyframe poses
  - current pose and trajectory
- The GUI must **only** ever see Snapshots, never live SLAM objects.

## Algorithm specification

### 1. Feature extraction (`features.py`)

- Use ORB with `nfeatures=1000`, `scaleFactor=1.2`, `nlevels=8`, `fastThreshold=20`, with a fallback of 7 for low-texture cells.
- **Grid distribution** (a simplified version of ORB-SLAM's quadtree):
  - Split the image into a grid of about 30 px cells per pyramid level, or approximate this at level 0.
  - Detect FAST per cell. If a cell finds nothing, retry with the low threshold.
  - Keep the strongest responses per cell until the per-level budget is met.
  - Then compute descriptors.
- Return keypoints, descriptors, and octave per keypoint.
- **Acceptance:** features spread visibly across the whole image, not clumped on the highest-contrast object.

### 2. Map initialization (`initializer.py`)

Simplified from ORB-SLAM3 `TwoViewReconstruction`. Implement the essential-matrix path only, and note in the docstring that ORB-SLAM3 also scores a homography model for planar scenes.

1. Choose a reference frame with at least 100 features.
2. For each new frame, match against the reference: brute-force Hamming, ratio test 0.8, cross-check, max distance 50. If there are fewer than 100 matches, keep waiting. Replace the reference if matches drop below 50.
3. Estimate the essential matrix: `findEssentialMat` with RANSAC, prob 0.999, threshold 1.0 px, then `recoverPose`.
4. Triangulate the inliers and keep a point only if:
   - its depth is positive in both views,
   - its reprojection error passes the χ² test in both views,
   - and its parallax angle is ≥ 1°.
5. Accept the initialization only if at least 50 points survive **and** the median parallax is ≥ 1°. Otherwise keep trying with later frames.
6. Create two keyframes and the map points. Normalize so median depth = 1. Run full BA over both keyframes and all points, with the first keyframe fixed.
7. Log a clear message on success or failure, with the reason (too few matches, low parallax, etc.).

### 3. Tracking (`tracking.py`)

**States:** `NO_IMAGES → NOT_INITIALIZED → OK ⇄ LOST`.

Per frame in state `OK`:

1. **Predict the pose** with a constant-velocity model: `V = T_cw_last · T_cw_prev⁻¹` (the relative motion between the last two frames), then `T_cw_pred = V · T_cw_last` (that motion applied again).
2. **Match by projection** (`matcher.search_by_projection`):
   - Project the last frame's map points into the current frame.
   - Search within a radius of 15 px × the octave scale. Accept the best match with Hamming distance ≤ 50 that is also clearly better than the second best (ratio 0.9).
   - If fewer than 20 matches, retry with radius 30.
   - If still fewer than 20, fall back to brute-force matching against the reference keyframe's map points.
3. **Pose-only optimization** (`optimizer.optimize_pose`):
   - Minimize the weighted reprojection error over the 6 pose parameters with `least_squares` and a Huber loss.
   - Run 4 rounds. After each round, mark χ² outliers and re-include inliers, as ORB-SLAM does.
4. **Track the local map:**
   - Local keyframes = keyframes sharing points with the current frame, plus their top covisible neighbors, capped at 10.
   - Project their map points that are not yet matched, then match and re-optimize the pose.
   - Increment `n_visible` and `n_found` on the map points.
5. **Decide success:** tracking succeeds if there are ≥ 30 inlier map points. Otherwise the state becomes `LOST`.
6. **Decide on a keyframe:** insert a new keyframe when:
   - at least `min_frames_between_kf` (default 5) have passed, and
   - either the tracked inliers are fewer than 90% of the reference keyframe's tracked points (and ≥ 15), or `max_frames_between_kf` (default 30) have passed.

**`LOST` handling (relocalization):**
- For each keyframe, do brute-force matching plus `solvePnPRansac` (reprojection threshold 4 px), followed by pose-only optimization.
- If ≥ 30 inliers, return to `OK`.
- If the map has fewer than 5 keyframes and tracking has been lost for more than 30 frames, reset the whole system to `NOT_INITIALIZED`.

### 4. Local mapping (`local_mapping.py`)

Runs synchronously on keyframe insertion in v1. Note in the docstring that ORB-SLAM3 runs this in its own thread.

1. Add observations for the new keyframe's matched map points. Update descriptors, viewing directions, and covisibility. Covisibility edges need ≥ 15 shared points.
2. **Cull recent map points.** Remove a point if:
   - its `n_found / n_visible` is below 0.25, or
   - two keyframes have passed since its creation and it is observed by fewer than 2 keyframes.
3. **Triangulate new points** between the new keyframe and its top covisible keyframes (up to 10):
   - Skip pairs whose baseline-to-median-depth ratio is below 0.01.
   - Match unassociated features by descriptor, constrained by the epipolar line (distance below 3σ px).
   - Triangulate, then apply the same depth, parallax, and χ² checks as initialization.
4. **Local BA** (`optimizer.local_bundle_adjustment`):
   - Optimize the new keyframe plus its covisible keyframes (up to 10) and every map point they observe.
   - Hold fixed any other keyframes that observe those points. Always hold the first keyframe fixed (gauge freedom).
   - Use `least_squares` with `jac_sparsity`, Huber loss, and `x_scale='jac'`, with a max of about 20 iterations.
   - Afterwards, remove observations that fail the χ² test, and mark points with fewer than 2 observations as bad.

### 5. Optimizer notes (`optimizer.py`)

- Write a single function for the reprojection residual and reuse it for pose-only optimization, local BA, and initialization BA.
- Build the Jacobian sparsity pattern explicitly. Each residual pair depends on 6 pose parameters and 3 point parameters. The function needs a docstring explaining why sparsity makes BA tractable.
- Log the initial and final cost and the time taken for every optimization call.

## GUI specification (`gui/`)

**Architecture:**
- One `QMainWindow`. SLAM runs in a `QThread` worker that pulls frames from the `FrameSource`, calls `SlamSystem.process`, and emits a `Snapshot` through a Qt signal.
- The GUI thread renders the latest snapshot and drops stale ones; it must never block the worker.

**Layout:** the window is split side by side, with a status bar.

**Left pane: video overlay (`overlay.py`).** Draw on the frame with OpenCV, then convert it to a `QImage`.
- **Keypoint colors:**
  - green circle: matched to a map point and an inlier (a landmark in use)
  - red: matched but rejected as an outlier
  - small grey dot: unmatched feature
- **Circle radius** scales with octave.
- **Optional toggle:** a short track line from each landmark's projected position in the last frame.
- **During initialization:** draw lines between reference and current matches.
- **HUD text:**
  - tracking state
  - FPS and per-stage ms (extract / track / local map)
  - number of features, tracked inliers, keyframes, and map points

**Right pane: 3D map (`map_view.py`, `pyqtgraph.opengl`).**
- **Map points:** a scatter plot. Points in the current local map are red; all others are light grey.
- **Keyframes:** small blue camera frustums drawn as line segments.
- **Current camera:** a green frustum.
- **Trajectory:** a line through the camera centers.
- **View controls:** a "follow camera" toggle, plus mouse orbit and zoom (built in).
- **Redraw rate:** the map view may redraw at a lower rate (about 5 Hz) than the video (up to 30 Hz).

**Keyboard shortcuts:**

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Reset SLAM |
| `T` | Toggle track lines |
| `F` | Toggle follow camera |
| `S` | Save trajectory (TUM format) and map points (PLY) to `output/<timestamp>/` |
| `Q` / `Esc` | Quit |

**Command line:**
```
python run.py --source 0                         # webcam index
python run.py --source path\to\video.mp4         # video file
python run.py --source path\to\tum_sequence      # image folder with rgb.txt
             --camera config\camera.yaml --config config\default.yaml
             [--headless] [--max-frames N]
```
`--headless` runs without the GUI and saves outputs at the end. This mode is used for tests and evaluation.

## Camera input and calibration

**`source.py`:**
- Webcams open with `cv2.VideoCapture(index, cv2.CAP_DSHOW)`, falling back to `CAP_MSMF`.
- Request the configured resolution (default 640×480) and verify what the camera actually delivers.
- Try to disable autofocus (`CAP_PROP_AUTOFOCUS`). Many webcams ignore this, so log whether the request took effect.
- Auto-exposure (`CAP_PROP_AUTO_EXPOSURE`) is explicitly switched **on** by default (changed with the owner's approval). Locking it is optional through config. The camera keeps the mode after the program exits, and manual exposure left the owner's laptop webcam near-black, so if it is locked, switch it back to auto on close.
- Timestamps come from `time.perf_counter()` for live input and from file timestamps for datasets.

**`tools/calibrate.py`:**
- Takes a printed checkerboard; inner corners and square size come from command-line arguments, defaulting to 9×6 and 25 mm.
- Shows the live feed, auto-captures a view when the board is detected and the pose differs enough from previous captures, and stops after 20 or more views.
- Runs `findChessboardCorners`, then `cornerSubPix`, then `calibrateCamera`.
- Reports the RMS reprojection error (warn if above 1.0 px) and writes `config/camera.yaml`.
- It must calibrate at the same resolution SLAM will use.

**Uncalibrated fallback:** if no `camera.yaml` exists, use `fx = fy = width / (2·tan(HFOV/2))` with an HFOV of 70° and zero distortion, and show a persistent warning in the HUD.

**`tools/record.py`:** records the webcam to `.mp4` plus a timestamps file, so the same run can be replayed while debugging.

## Testing

**Synthetic tests** (`tests/synthetic.py`): generate random 3D points and a known camera trajectory, project with pixel noise (σ = 0.5–1 px), and add outlier matches.

| Test | Pass criterion |
|---|---|
| Two-view initialization | Recovered relative rotation within 0.5° and translation direction within 2° of ground truth |
| Pose-only optimization | Converges from a perturbed pose (5°, 10% of depth) to within 0.2° / 1% with 20% outliers |
| Bundle adjustment | Final cost lower than initial; poses and points closer to ground truth after a similarity alignment |
| Geometry helpers | SE(3) inverse/compose round-trips; triangulation of noise-free points is exact to 1e-6 |

**Dataset test** (manual, documented in the README):
- Data: the TUM RGB-D `fr1_xyz` and `fr1_desk` sequences, using RGB images only, with TUM's published intrinsics in a `camera.yaml`.
- Run headless, then evaluate with `evo_ape tum groundtruth.txt estimate.txt -as` (the `-as` flags align and correct scale, as required for monocular).
- Target: scale-aligned ATE RMSE under about 5 cm on `fr1_xyz` and no tracking loss. Full ORB-SLAM gets roughly 1 cm, so this is a "working, not tuned" bar. Report the result in the README.

**Live test:** with the webcam, initialize by **translating** the camera sideways about 10–20 cm. Pure rotation cannot initialize a monocular system. Then walk a loop around a desk. Tracking should stay `OK` at moderate speed, and the 3D map should show recognizable structure.

## Milestones

1. **Skeleton + capture + GUI shell.**
   - Done when: the repo layout, config loading, and `FrameSource` are in place, and the GUI shows live webcam video with FPS. The 3D view shows axes only.
2. **Calibration.**
   - Done when: `tools/calibrate.py` works, `camera.yaml` is written, and the RMS error is reported.
3. **Features + overlay.**
   - Done when: grid-distributed ORB features are drawn live and per-stage timing appears in the HUD.
4. **Initialization.**
   - Done when: synthetic init tests pass, and live, translating the camera produces the first two keyframes and a point cloud in the 3D view.
5. **Tracking.**
   - Done when: synthetic pose tests pass, and live, the overlay shows green landmarks tracked frame to frame and the camera frustum moves in 3D.
6. **Local mapping + local BA.**
   - Done when: BA tests pass, keyframes are added during motion, the map grows, and scale stays roughly consistent over about a 1 m loop.
7. **Relocalization + reset + saving.**
   - Done when: covering the lens then returning to a mapped view recovers tracking, and `S` writes valid TUM and PLY files.
8. **Evaluation + README.**
   - Done when: the TUM `fr1_xyz` results are reported and the README covers install, calibration, run, operating tips, and known limitations.

## Performance target

At 640×480 with 1000 features, aim for at least 10 FPS of tracking on a typical laptop CPU. Local BA may cause brief stalls on keyframe insertion, which is acceptable in v1.

If performance falls well short, profile it (`cProfile`) and report the hotspots to the owner before optimizing. Do not restructure the code for speed without asking.

## Known limitations to document in the README

- Monocular scale is arbitrary and drifts. No loop closure is implemented.
- Laptop webcams have rolling shutters, motion blur, and autofocus/auto-exposure that change over time. Move slowly and keep lighting steady.
- Low-texture views (blank walls) and dynamic objects (people) cause tracking loss.
- Pointers to the next steps toward the blimp use case:
  - visual-inertial fusion
  - loop closure
  - multi-camera rigs
  - full ORB-SLAM3 / OpenVINS for comparison
