"""
Brain Visualizer — Renders TRIBE v2 predictions onto a 3D brain surface.

Uses nilearn + matplotlib for rendering (no OpenGL/PyVista needed).

Takes the raw (n_timesteps, 20484) prediction array and produces:
  - Static brain images (mean activation, peak activation)
  - Per-timestep frame PNGs
  - An MP4 movie of brain activity over time

The 20,484 vertices map to fsaverage5:
  - vertices 0:10242     = left hemisphere
  - vertices 10242:20484 = right hemisphere
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window needed
import matplotlib.pyplot as plt
import numpy as np


def _get_fsaverage5_meshes():
    """Load the fsaverage5 surface meshes from nilearn."""
    from nilearn import datasets
    return datasets.fetch_surf_fsaverage(mesh="fsaverage5")


def _get_clim(preds: np.ndarray):
    """Compute robust symmetric color limits using 2nd-98th percentile."""
    vmin = np.percentile(preds, 2)
    vmax = np.percentile(preds, 98)
    return max(abs(vmin), abs(vmax))


def _render_brain_figure(data_1d: np.ndarray, fsaverage, abs_max: float,
                         title: str = "", subtitle: str = "",
                         surface_type: str = "infl"):
    """
    Render a single brain activation map as a matplotlib figure.
    Shows 4 views: left lateral, left medial, right lateral, right medial.
    """
    from nilearn import plotting

    # Split hemispheres
    lh_data = data_1d[:10242]
    rh_data = data_1d[10242:]

    # Pick surface mesh
    surf_key_map = {"infl": "infl", "inflated": "infl", "pial": "pial", "white": "white"}
    surf_key = surf_key_map.get(surface_type, "infl")
    lh_mesh = fsaverage[f"{surf_key}_left"]
    rh_mesh = fsaverage[f"{surf_key}_right"]
    lh_bg = fsaverage["sulc_left"]
    rh_bg = fsaverage["sulc_right"]

    fig = plt.figure(figsize=(16, 12), facecolor="white")

    # ── Title + subtitle ──
    fig.text(0.5, 0.97, title, fontsize=24, fontweight="bold",
             ha="center", va="top", color="black")
    if subtitle:
        fig.text(0.5, 0.94, subtitle, fontsize=14, ha="center", va="top",
                 color="black")

    # ── Brain views ──
    margin_x = 0.02
    gap_x = 0.02
    col_w = (1.0 - 2 * margin_x - gap_x) / 2

    top_row_bottom = 0.50
    bot_row_bottom = 0.16
    row_h = 0.40

    positions = [
        [margin_x,                    top_row_bottom, col_w, row_h],
        [margin_x + col_w + gap_x,    top_row_bottom, col_w, row_h],
        [margin_x,                    bot_row_bottom, col_w, row_h],
        [margin_x + col_w + gap_x,    bot_row_bottom, col_w, row_h],
    ]

    view_configs = [
        (lh_mesh, lh_data, lh_bg, "left",  "lateral"),
        (rh_mesh, rh_data, rh_bg, "right", "lateral"),
        (lh_mesh, lh_data, lh_bg, "left",  "medial"),
        (rh_mesh, rh_data, rh_bg, "right", "medial"),
    ]

    view_labels = ["Left Lateral", "Right Lateral", "Left Medial", "Right Medial"]

    for pos, (mesh, data, bg_map, hemi, view), label in zip(positions, view_configs, view_labels):
        # Label right at the top edge of the brain box — sits on the brain
        fig.text(pos[0] + pos[2] / 2, pos[1] + pos[3] - 0.04, label,
                 fontsize=13, fontweight="bold", ha="center", va="top",
                 color="black")

        ax = fig.add_axes(pos, projection="3d")
        plotting.plot_surf_stat_map(
            mesh, data,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            colorbar=False,
            cmap="RdBu_r",
            vmax=abs_max,
            threshold=abs_max * 0.05,
            axes=ax,
        )

    # ── Colorbar — pushed up higher with more room ──
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.025])
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-abs_max, vmax=abs_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Predicted fMRI Activation (Below Baseline ← 0 → Active)",
                   fontsize=13, fontweight="bold", labelpad=10, color="black")
    cbar.ax.tick_params(labelsize=11)

    return fig


def render_mean_brain(preds: np.ndarray, output_path: str, surface: str = "inflated"):
    """Render the mean activation across all timesteps and save as PNG."""
    print("    Rendering mean brain...")
    fsaverage = _get_fsaverage5_meshes()
    abs_max = _get_clim(preds)
    mean_data = preds.mean(axis=0)

    stats = (f"Averaged across {preds.shape[0]} timestep(s)  |  "
             f"Range: [{mean_data.min():.4f}, {mean_data.max():.4f}]  |  "
             f"Mean: {mean_data.mean():.4f}")

    fig = _render_brain_figure(mean_data, fsaverage, abs_max,
                                title="Mean Brain Activation",
                                subtitle=stats,
                                surface_type=surface)
    fig.savefig(output_path, dpi=150, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def render_peak_brain(preds: np.ndarray, output_path: str, surface: str = "inflated"):
    """Render the timestep with the highest absolute activation."""
    # Find the timestep with the strongest overall activation
    abs_means = np.abs(preds).mean(axis=1)
    peak_t = int(np.argmax(abs_means))

    print(f"    Rendering peak brain (timestep {peak_t})...")
    fsaverage = _get_fsaverage5_meshes()
    abs_max = _get_clim(preds)
    peak_data = preds[peak_t]

    stats = (f"Timestep {peak_t} (strongest activation)  |  "
             f"Range: [{peak_data.min():.4f}, {peak_data.max():.4f}]  |  "
             f"Mean: {peak_data.mean():.4f}")

    fig = _render_brain_figure(peak_data, fsaverage, abs_max,
                                title=f"Peak Brain Activation (t={peak_t})",
                                subtitle=stats,
                                surface_type=surface)
    fig.savefig(output_path, dpi=150, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def _render_brain_frame_fast(data_1d: np.ndarray, fsaverage, abs_max: float,
                             title: str = "", surface_type: str = "infl"):
    """Fast frame renderer — 2 lateral views only, small figure. ~60% faster."""
    from nilearn import plotting

    lh_data = data_1d[:10242]
    rh_data = data_1d[10242:]

    surf_key_map = {"infl": "infl", "inflated": "infl", "pial": "pial", "white": "white"}
    surf_key = surf_key_map.get(surface_type, "infl")
    lh_mesh = fsaverage[f"{surf_key}_left"]
    rh_mesh = fsaverage[f"{surf_key}_right"]
    lh_bg = fsaverage["sulc_left"]
    rh_bg = fsaverage["sulc_right"]

    fig = plt.figure(figsize=(10, 5), facecolor="white")
    if title:
        fig.text(0.5, 0.95, title, fontsize=16, fontweight="bold",
                 ha="center", va="top", color="black")

    views = [
        ([0.0, 0.05, 0.5, 0.85], lh_mesh, lh_data, lh_bg, "left", "lateral"),
        ([0.5, 0.05, 0.5, 0.85], rh_mesh, rh_data, rh_bg, "right", "lateral"),
    ]
    for pos, mesh, data, bg_map, hemi, view in views:
        ax = fig.add_axes(pos, projection="3d")
        plotting.plot_surf_stat_map(
            mesh, data, hemi=hemi, view=view, bg_map=bg_map,
            colorbar=False, cmap="RdBu_r", vmax=abs_max,
            threshold=abs_max * 0.05, axes=ax,
        )
    return fig


def render_frames(preds: np.ndarray, frames_dir: str, surface: str = "inflated",
                  fast: bool = True):
    """Render each timestep as a separate PNG frame.

    Args:
        fast: If True, render 2 lateral views only (much faster for movies).
              If False, render full 4-view figure.
    """
    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)

    fsaverage = _get_fsaverage5_meshes()
    abs_max = _get_clim(preds)
    n_times = preds.shape[0]

    print(f"    Rendering {n_times} frames ({'fast 2-view' if fast else 'full 4-view'})...")
    frame_files = []
    for t in range(n_times):
        data = preds[t]

        if fast:
            fig = _render_brain_frame_fast(data, fsaverage, abs_max,
                                           title=f"t={t}s", surface_type=surface)
        else:
            stats = (f"Range: [{data.min():.4f}, {data.max():.4f}]  |  "
                     f"Mean: {data.mean():.4f}")
            fig = _render_brain_figure(data, fsaverage, abs_max,
                                        title=f"Brain Activity — t={t}s",
                                        subtitle=stats, surface_type=surface)

        frame_path = str(frames_path / f"frame_{t:03d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches=None, facecolor="white")
        plt.close(fig)
        frame_files.append(frame_path)
        print(f"      Frame {t + 1}/{n_times} saved")

    print(f"    All frames saved to: {frames_dir}")
    return frame_files


def render_movie(preds: np.ndarray, output_path: str, fps: int = 2,
                 surface: str = "inflated", frames_dir: str = None):
    """
    Render a movie of brain activity across timesteps.
    Renders each timestep as a PNG frame, stitches into video with OpenCV.
    """
    import tempfile

    save_dir = frames_dir if frames_dir else tempfile.mkdtemp(prefix="brain_frames_")
    frame_files = render_frames(preds, save_dir, surface)

    if not frame_files:
        print("    No frames to stitch into movie.")
        return

    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print("    Error: Could not read rendered frames.")
        return

    h, w = first_frame.shape[:2]
    # OpenCV needs even dimensions
    h = h - (h % 2)
    w = w - (w % 2)

    avi_path = str(Path(output_path).with_suffix(".avi"))
    print(f"    Stitching {len(frame_files)} frames into movie ({fps} fps)...")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        # Fallback codec if XVID not available
        print("    XVID codec not available, trying MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        avi_path = str(Path(output_path).with_suffix(".avi"))
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print("    Error: Could not open video writer. Frames saved as PNGs instead.")
        return

    for fpath in frame_files:
        frame = cv2.imread(fpath)
        if frame is not None:
            # Resize to even dimensions if needed
            frame = cv2.resize(frame, (w, h))
            writer.write(frame)

    writer.release()

    # Verify the file was actually written
    avi_file = Path(avi_path)
    if avi_file.exists() and avi_file.stat().st_size > 0:
        print(f"    Saved: {avi_path}")
    else:
        print("    Error: Video file is empty. Frames saved as PNGs instead.")

    # Clean up temp frames if we didn't want to keep them
    if not frames_dir:
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)


def render_html(preds: np.ndarray, output_path: str, surface: str = "inflated"):
    """Render an interactive HTML brain viewer using nilearn's WebGL engine.

    Near-instant — no matplotlib 3D rendering. Opens in any browser.
    """
    from nilearn import plotting, datasets

    fsaverage = _get_fsaverage5_meshes()
    mean_data = preds.mean(axis=0)
    abs_max = _get_clim(preds)

    surf_key_map = {"inflated": "infl", "pial": "pial", "white": "white"}
    surf_key = surf_key_map.get(surface, "infl")

    view = plotting.view_surf(
        fsaverage[f"{surf_key}_left"],
        mean_data[:10242],
        cmap="RdBu_r",
        symmetric_cmap=True,
        vmax=abs_max,
        bg_map=fsaverage["sulc_left"],
        title="TRIBE v2 — Mean Brain Activation (Left Hemisphere)",
    )
    view.save_as_html(output_path)
    print(f"    Saved interactive HTML: {output_path}")


def visualize_all(preds: np.ndarray, output_dir: str, image_stem: str,
                  surface: str = "inflated", save_frames: bool = False,
                  make_movie: bool = True, movie_fps: int = 2,
                  html: bool = False):
    """Run the full visualization pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n  BRAIN VISUALIZATION")
    print("  " + "-" * 50)

    # Interactive HTML viewer (near-instant)
    if html:
        try:
            render_html(preds, str(out / f"{image_stem}_brain.html"), surface)
        except Exception as e:
            print(f"    HTML render failed: {e}")

    try:
        render_mean_brain(preds, str(out / f"{image_stem}_mean_brain.png"), surface)
    except Exception as e:
        print(f"    Mean brain render failed: {e}")

    try:
        render_peak_brain(preds, str(out / f"{image_stem}_peak_brain.png"), surface)
    except Exception as e:
        print(f"    Peak brain render failed: {e}")

    if make_movie:
        frames_dir = str(out / "frames") if save_frames else None
        try:
            render_movie(
                preds,
                str(out / f"{image_stem}_brain_movie.avi"),
                fps=movie_fps,
                surface=surface,
                frames_dir=frames_dir,
            )
        except Exception as e:
            print(f"    Movie render failed: {e}")

    elif save_frames:
        try:
            render_frames(preds, str(out / "frames"), surface)
        except Exception as e:
            print(f"    Frame render failed: {e}")

    print("  Visualization complete.")
