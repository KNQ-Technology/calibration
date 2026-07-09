# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monocular camera calibration toolkit for soccer/training-field scenes (part of the larger MCS system). It maps pixel coordinates in video frames to physical pitch coordinates, with lens distortion correction, marker measurement, and visualization. Code comments, console output, and docs (`README.md`, `doc_lark/`) are in Chinese.

## Commands

README says Python 3.10, but on this machine the project dependencies are installed under **Python 3.13** (`C:\Users\24590\AppData\Local\Programs\Python\Python313\python.exe`). The default `python` on PATH is 3.11 and lacks `supervision` etc. — always launch project scripts with the 3.13 executable.

```bash
pip install -r requirements.txt

# Main calibration + marker measurement pipeline (interactive)
python measure_marker_position.py

# Tkinter GUI annotation client (alternative to the OpenCV annotation flow)
python annotation_client.py
```

There are no tests or linters. The main script opens interactive OpenCV/tkinter windows and prompts on stdin, so it cannot run headless.

## Configuration

`config.yaml` at the repo root drives everything:
- `video_name` / `CAMERA_NAME`: input video and the storage folder name.
- `config`: the name of a pitch-configuration class in `soccer.py` (e.g. `QingYuanConfiguration`), resolved via `getattr(soccer, name)()`. Each class defines pitch dimensions (in **cm**), keypoint `vertices`, `edges`, and `labels`.
- `GROUP_CONFIGS` / `PAIR_CONFIGS`: which within-group and between-group marker distances to compute.
- `jac` (`2-point`/`3-point` finite differences), `reprojection_eps`, `enable_orientation`: optimizer settings read directly from `config.yaml` by `de_distortion.py` and `check_valid.py` (not passed through `main`). Per the README, delete `store/{CAMERA_NAME}/distortion_params.json` before comparison experiments so distortion is recomputed.

Note: `config.py` is unrelated to `config.yaml` — it holds path constants for an optional BoT-SORT/YOLO detection setup (`models/`, `output/`, gitignored) used by `tools/single_camera_tracker.py`.

## Architecture

**Pipeline** (`measure_marker_position.py` → `main()`): load `config.yaml` → construct `Camera` → annotate/load marker groups → `pixel_to_pitch` conversion → distance calculations → report + visualization saved to `store/manual/{CAMERA_NAME}/`.

**`camera.py`** is the core (~3000 lines):
- `Camera.__init__` runs the full calibration on construction: reads a frame (index `IDX` from `distort.py`), calibrates keypoints and distortion, builds the homography, computes FOV/visible grids via `CameraFOVCalculator`, sets up `TrainArea`, and writes visualizations — all cached under `calibration_dir` (default `store/{CAMERA_NAME}/`, or `store/{mcs_name}_mcs/{camera_name}/`).
- Coordinate transform chain: pixel → undistort → homography → pitch coords. Use `camera.pixel_to_pitch()` / `pitch_to_pixel()`; distortion is applied automatically.
- Calibration files are cached; missing files trigger interactive annotation. `debug_mode=True` forces recomputation of distortion params and camera_info but still saves results.
- Also contains `Detection`/`CameraDetection` (YOLO detection, optional — init failure is caught) and `TrainArea`.

**Distortion** (`de_distortion.py`): Brown-Conrady 5-parameter model (k1,k2,k3,p1,p2) fitted with `scipy.optimize.least_squares` (LM) against annotated straight lines (`manual_horizontal.json` / `manual_vertical.json`). Sign convention: correction params = −camera params; the `distort_points`/`undistort_points` wrappers handle the conversion. `distort.py` has the raw forward/iterative-inverse formulas and the `IDX` constant (which video frame is used for calibration).

**Annotation** (`area.py`): OpenCV-window annotators — `KeypointAnnotator` for pitch keypoints (matched by vertex index to the soccer config), `MarkerAnnotator` for multi-group marker points. `annotation_client.py` is a standalone tkinter GUI covering the same JSON formats.

**Support**: `grid_system.py` (`PitchGridSystem`, the grid definition shared with MCS), `check_valid.py` (reprojection/orientation validity checks used during optimization), `tools/` (pitch drawing, distance calculation/drawing, coordinate transforms, frame I/O, BoT-SORT tracker).

## Conventions & Gotchas

- **Units**: pitch/physical coordinates are in **cm** throughout; measurement reports convert to meters (divide by 100) at display time.
- **Chinese file paths**: video and store folders contain Chinese characters. `cv2.imwrite`/`cv2.imread` fail on these on Windows — use `imwritex` from `area.py` (which uses `cv2.imencode` + `tofile`).
- **Calibration cache layout** in `store/{CAMERA_NAME}/`: `keypoints.json`, `manual_horizontal.json`, `manual_vertical.json`, `distortion_params.json`, `train_area.json`, `camera_info.json`, plus `*_annotated.png` / visualization images. Marker annotations and measurement outputs go to `store/manual/{CAMERA_NAME}/` (`markers.json`, `coordinates.json`, `measurement_report.txt`, `visualization.jpg`).
- Constructing a `Camera` has heavy side effects (file I/O, possible interactive annotation, visualization writes); keep that in mind when adding code paths that create one.
