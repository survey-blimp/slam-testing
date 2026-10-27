## slam-testing

This repo is for the 2703 Capstone team to test and validate SLAM for the survey blimp.

It contains **minislam**, a small, readable monocular ORB-SLAM in Python. It follows the structure of ORB-SLAM3 (ORB features, map points, keyframes, covisibility graph, local bundle adjustment), implements only the core, and runs on a laptop webcam on Windows. A Qt GUI shows the live video with the landmarks in use next to a 3D view of the map and the camera trajectory.

This is a learning implementation: clarity beats speed. Every module starts with a docstring explaining the math and naming the ORB-SLAM3 file it corresponds to. Every OpenCV call that hides real algorithmic work is marked `# BLACKBOX:` with a note on what a by-hand version would involve. The full specification is in [AGENTS.md](AGENTS.md).

## Install

Requires Windows 10/11 and Python 3.13. The spec originally said 3.11; it was changed to 3.13 with the owner's approval, and every dependency ships 3.13 wheels.

```bash
py -3.13 -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Check the install:

```bash
python -m pytest
```

The tests take about a minute. They include an end-to-end run of the whole system on rendered images (see [Synthetic test sequence](#synthetic-test-sequence)).

## Calibrate your camera

Without calibration minislam guesses the intrinsics from a 70° horizontal field of view and shows a red **UNCALIBRATED** warning in the HUD. That works, but calibrating is worth the five minutes:

1. Print a checkerboard. The default is **9×6 inner corners** (10×7 squares) with **25 mm** squares. Measure a printed square with a ruler; printers scale.
2. Tape it to something flat and rigid.
3. Run the calibration tool and show the board to the camera from many angles: tilted, close, far, and in the corners of the image. A view is captured automatically whenever the board is detected in a new enough position. It stops after 20 views.

```bash
python tools/calibrate.py --cols 9 --rows 6 --square-mm 25
```

This writes `config/camera.yaml` and prints the RMS reprojection error. Below about 0.5 px is good; above 1.0 px it warns (usual causes are blur, a board that isn't flat, or the wrong square size or corner count). It captures at the resolution in `config/default.yaml` (`source.width`/`height`, 640×480 by default), the same resolution SLAM uses. Calibrate again whenever you change it.

You can also calibrate from a folder of saved photos with `--images path\to\folder`.

## Run

```bash
python run.py --source 0
```

| Command | Input |
|---|---|
| `python run.py --source 0` | webcam index 0 |
| `python run.py --source recordings\desk.mp4` | a video file (made with `tools/record.py`, or any video) |
| `python run.py --source data\rgbd_dataset_freiburg1_xyz` | an image folder with a TUM-style `rgb.txt` |

Other options: `--camera config\camera.yaml` (the default), `--config config\default.yaml` (the default), `--headless` (no GUI; saves outputs at the end), `--max-frames N`, `--output DIR` (output folder for `--headless`), and `--log-level DEBUG` (also logs every pose optimization).

### Keyboard

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Reset SLAM |
| `T` | Toggle track lines |
| `F` | Toggle follow camera in the 3D view |
| `S` | Save trajectory (TUM) and map points (PLY) to `output/<timestamp>/` |
| `Q` / `Esc` | Quit |

In the 3D view, left-drag orbits, middle-drag pans, and the wheel zooms.

### What you see

**Video (left)**
- **Green circle:** a feature matched to a map point and accepted, i.e. a landmark in use. The radius grows with the pyramid octave.
- **Red circle:** a feature matched to a map point but rejected as an outlier.
- **Grey dot:** a feature with no map point.
- **Yellow lines (initialization only):** matches between the reference frame and the current frame.
- **Orange lines (`T`):** where each landmark projected in the previous frame.
- **HUD:** tracking state, FPS, per-stage times (ORB extraction / tracking / local mapping), and counts of features, tracked inliers, keyframes and map points. Below that, the latest status message, such as why initialization is still waiting.

**3D map (right)**
- **Points:** red if in the current local map, light grey otherwise.
- **Blue frustums:** keyframes.
- **Green frustum:** the current camera.
- **Yellow line:** the trajectory.
- **Axes:** the world origin, which is the first keyframe.

The display draws SLAM coordinates (x right, y down, z forward) as (x, z, −y), so "up" looks up.

### Record and replay

```bash
python tools/record.py --output recordings\desk.mp4
```

This writes `desk.mp4` and `desk.timestamps.txt`. Running `run.py --source recordings\desk.mp4` uses the timestamps automatically, so you can replay a live session exactly while debugging.

## Operating tips

- **Initialize by translating, not rotating.** Point the camera at a textured scene and slide it sideways about 10–20 cm. A monocular camera can't measure depth from a pure rotation; the HUD tells you to translate more until the parallax is big enough.
- **Move slowly and smoothly**, especially when turning. Webcams have rolling shutters and motion blur.
- **Keep the lighting steady** and aim at texture: desks, shelves, posters. Avoid blank walls, windows and moving people.
- **If tracking is lost**, go back to a view you have already mapped and hold still; relocalization tries every keyframe. If the map is still tiny (under 5 keyframes) and you stay lost for more than 30 frames, the map resets and initialization starts over. `R` resets at any time.
- **Autofocus lock** is requested at start-up. Many webcams ignore it, and the log says whether it took effect. The built-in webcam on this machine doesn't support it (it is probably fixed-focus).
- **Auto-exposure is switched on** at start-up (`source.disable_auto_exposure: false`). Locking it puts the camera in manual mode at the driver's default exposure, which on the Legion 5 webcam is almost black indoors. The camera also *remembers* the mode after the program exits. If you lock it anyway, minislam switches auto back on when it closes. If the image is ever dark from an earlier run, just start minislam (or open and close the Windows Camera app).
- **Turn off Windows Studio Effects** (Settings → Bluetooth & devices → Cameras → your camera, or the *Studio effects* tile in Quick Settings). *Automatic framing* digitally crops and zooms to follow faces, which changes the focal length and image centre from frame to frame and breaks tracking. *Background blur* removes background features, and *eye contact* warps the image. These effects run inside the camera driver, so OpenCV sees them too. Recalibrate after turning them off.

## Monocular scale

A single camera can't observe absolute scale. minislam picks a scale at initialization: **the median depth of the first map is 1 unit**. All positions, the saved trajectory and the PLY are in those arbitrary units. Without loop closure, the scale also slowly **drifts** over long runs. Comparisons with ground truth therefore always use a similarity (scale-correcting) alignment, such as `evo -as`.

## Outputs

`S` (or the end of a `--headless` run) writes:

| File | Contents |
|---|---|
| `trajectory.txt` | Every tracked frame in TUM format (`timestamp tx ty tz qx qy qz qw`, camera to world). Each pose is stored relative to its reference keyframe, so it also benefits from later bundle adjustment. |
| `keyframes.txt` | The keyframe poses in the same format |
| `map_points.ply` | The map points, viewable in MeshLab or CloudCompare |

## Evaluation

### TUM RGB-D fr1_xyz (manual)

1. Download `rgbd_dataset_freiburg1_xyz.tgz` (and optionally `fr1_desk`) from the [TUM RGB-D dataset download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and extract it into `data\`. `data\` is git-ignored.
2. Run headless with TUM's published intrinsics. Only the RGB images are used.

```bash
python run.py --source data\rgbd_dataset_freiburg1_xyz --camera config\camera_tum_fr1.yaml --headless --output output\fr1_xyz
```

3. Evaluate with evo (an optional dev dependency). `-as` aligns and corrects scale, which is required for monocular.

```bash
pip install evo
```

```bash
evo_ape tum data\rgbd_dataset_freiburg1_xyz\groundtruth.txt output\fr1_xyz\trajectory.txt -as --plot
```

The target is a scale-aligned ATE RMSE under about 5 cm with no tracking loss. Full ORB-SLAM gets about 1 cm, so this is a "working, not tuned" bar.

| Sequence | ATE RMSE (scale-aligned) | Tracking loss | Notes |
|---|---|---|---|
| fr1_xyz | *not run yet* | | |
| fr1_desk | *not run yet* | | |

### Synthetic test sequence

`tests/synthetic.py` can also ray-cast a textured room seen by a camera with a known trajectory: a 0.3 m sideways slide to initialize, then a loop of about 1 m × 0.8 m with gentle yaw. It writes the sequence in TUM layout, with `rgb.txt`, `groundtruth.txt` and a `camera.yaml`:

```bash
python tests\synthetic.py data\synthetic_room --frames 300
```

```bash
python run.py --source data\synthetic_room --camera data\synthetic_room\camera.yaml
```

The current results on this 300-frame sequence (3.2 m path) are:

| Metric | Result |
|---|---|
| Tracking loss after initialization | none |
| Initialization | at frame 9 (9 cm baseline) |
| Keyframes | 10 |
| ATE RMSE (scale-aligned) | 5.9 mm |
| Throughput | 13.2 FPS overall |
| Median frame | 54 ms |
| Local mapping per keyframe | about 0.4–1.2 s |

The scene is noise-free and perfectly textured, so this checks the pipeline, not real-world accuracy.

## Performance

At 640×480 with 1000 features, on the development laptop:

| Stage | Time |
|---|---|
| ORB extraction | about 20 ms |
| Tracking | 30–50 ms |
| Local mapping + local BA | 0.4–1.2 s per keyframe, during which tracking stalls (allowed in v1) |

Profiling (`python -m cProfile`) shows local BA as the hot spot, at about 60% of the time when keyframes are frequent. Within BA the cost splits between the sparse LSMR solve and the finite-difference Jacobian. Local BA also includes every keyframe that sees a local point as a fixed camera, and without keyframe culling that set keeps growing. The usual next steps, none implemented in v1, are:

- an analytic Jacobian,
- a Schur-complement solver,
- keyframe culling,
- running local mapping in its own thread, as ORB-SLAM3 does.

## Known limitations

- **Scale.** Monocular scale is arbitrary and drifts.
- **No loop closure.** There is no loop closure, place recognition (DBoW) or pose-graph optimization, so drift is never corrected. Relocalization brute-forces every keyframe instead of querying a vocabulary.
- **No map maintenance.** There is no keyframe culling and no fusion of duplicate map points. Long sessions slowly accumulate redundant keyframes and points, and local BA gets slower.
- **Essential matrix only.** Initialization uses only the essential matrix. ORB-SLAM3 also scores a homography, which handles planar or low-parallax scenes better.
- **Webcam limitations.** Laptop webcams have rolling shutters, motion blur, and autofocus and auto-exposure that change over time. Move slowly and keep the lighting steady.
- **Difficult scenes.** Low-texture views (blank walls) and dynamic objects (people) cause tracking loss.
- **Synchronous local mapping.** Local mapping runs inside the tracking thread, so keyframe insertion causes visible stalls.
- **Slow BA convergence from large errors.** scipy's `least_squares` (`trf`) with a Huber loss converges slowly when most residuals are large (several pixels). Within the running system BA starts from good estimates and converges in a few iterations.

### Next steps toward the blimp use case

- Visual-inertial fusion (IMU): metric scale, robustness to fast motion and texture-poor views.
- Loop closure, to correct drift over long survey flights.
- Multi-camera rigs, for a wider field of view and stereo scale.
- Comparison against full ORB-SLAM3 and OpenVINS on the same recordings.

## Code map

| Module | Role | ORB-SLAM3 equivalent |
|---|---|---|
| [minislam/config.py](minislam/config.py) | YAML → validated dataclasses | `Settings.cc` |
| [minislam/camera.py](minislam/camera.py) | pinhole model, distortion, keypoint undistortion | `CameraModels/Pinhole.cc` |
| [minislam/source.py](minislam/source.py) | webcam, video file and image-folder input | the example programs |
| [minislam/features.py](minislam/features.py) | ORB with pyramid, grid distribution, orientation | `ORBextractor.cc` |
| [minislam/frame.py](minislam/frame.py) | per-image features, pose, associations | `Frame.cc` |
| [minislam/map.py](minislam/map.py) | MapPoint, KeyFrame, Map, covisibility | `MapPoint.cc`, `KeyFrame.cc`, `Map.cc` |
| [minislam/matcher.py](minislam/matcher.py) | brute-force, projection and epipolar matching | `ORBmatcher.cc` |
| [minislam/geometry.py](minislam/geometry.py) | SE(3), triangulation and its checks, epipolar geometry | `GeometricTools.cc`, Sophus |
| [minislam/initializer.py](minislam/initializer.py) | two-view initialization | `TwoViewReconstruction.cc` |
| [minislam/tracking.py](minislam/tracking.py) | state machine, motion model, local map, relocalization | `Tracking.cc` |
| [minislam/local_mapping.py](minislam/local_mapping.py) | keyframe insertion, culling, triangulation, local BA | `LocalMapping.cc` |
| [minislam/optimizer.py](minislam/optimizer.py) | pose-only optimization and bundle adjustment | `Optimizer.cc` |
| [minislam/system.py](minislam/system.py) | wires everything together, builds Snapshots | `System.cc` |
| [minislam/snapshot.py](minislam/snapshot.py) | immutable state handed to the GUI | `FrameDrawer` / `MapDrawer` copies |
| [minislam/gui/](minislam/gui/) | Qt window, worker thread, overlay, 3D view | `Viewer.cc`, `FrameDrawer.cc`, `MapDrawer.cc` |

Every tunable number is in [config/default.yaml](config/default.yaml), each with a comment.
