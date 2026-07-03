#!/usr/bin/env python
"""Bake the docs gallery web assets from ``docs/USD.zip``.

Run locally once (the outputs are committed; ReadTheDocs never runs this):

    /insomnia001/depts/morpheus/users/ob2391/miniforge3/envs/waxmorph/bin/python \
        docs/scripts/bake_gallery.py

For each selected USD scene this reads the ``/root/cells`` ``UsdGeomPointInstancer``
with ``pxr`` and writes a compact, quantized, browser-friendly bundle under
``docs/_static/gallery/<slug>/``:

* ``positions.u16``  — uint16, ``nFrames * count * 3``, quantized against the
  per-scene bounding box (dequantized in JS with ``bbox`` from the manifest).
* ``colors.u8``      — uint8 palette indices, ``nFrames * count`` (per-cell
  ``displayColor`` snapped to a per-scene palette; colors evolve over time).
* ``manifest.json``  — geometry/animation metadata + the colour palette + the
  per-frame radius (cells share one radius per frame).
* ``poster.png``     — a matplotlib scatter of a late frame for the gallery card.

A top-level ``docs/_static/gallery/index.json`` lists the scenes grouped into the
three page sections. The scenes are the 16 curated in the plan.

USD facts this relies on (verified against the data, but re-checked defensively):
Y-up, single ``/root/cells`` PointInstancer, one ``Sphere`` prototype (radius 1),
identity ``/root`` transform (positions are world space), isotropic scale that is
uniform across cells within a frame, no per-instance orientation.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
USD_ZIP = DOCS / "USD.zip"
OUT_ROOT = DOCS / "_static" / "gallery"

# (usd filename, slug, section, label, family, organizer-tag)
SCENES = [
    # --- Featured: one base reconstruction per family ---
    ("sphere_bunny_matched.usd", "sphere-bunny", "featured", "Sphere → Bunny", "sphere_bunny", None),
    ("armadillo_bunny_matched.usd", "armadillo-bunny", "featured", "Armadillo → Bunny", "armadillo_bunny", None),
    ("mouse_limb_10.5_11.5_extentmatched.usd", "mouse-limb", "featured", "Mouse limb 10.5 → 11.5", "mouse_limb", None),
    ("heart_repeat_4.usd", "heart", "featured", "Heart", "heart", None),
    # --- Organizer sweep: armadillo ↔ bunny ---
    ("armadillo_bunny__uniform_no_cue__repeat_04.usd", "ab-org-nocue", "armadillo_bunny_organizers", "No cue", "armadillo_bunny", "no_cue"),
    ("armadillo_bunny__organizers_03__repeat_05.usd", "ab-org-03", "armadillo_bunny_organizers", "3 organizers", "armadillo_bunny", 3),
    ("armadillo_bunny__organizers_06__repeat_01.usd", "ab-org-06", "armadillo_bunny_organizers", "6 organizers", "armadillo_bunny", 6),
    ("armadillo_bunny__organizers_09__repeat_01.usd", "ab-org-09", "armadillo_bunny_organizers", "9 organizers", "armadillo_bunny", 9),
    ("armadillo_bunny__organizers_12__repeat_01.usd", "ab-org-12", "armadillo_bunny_organizers", "12 organizers", "armadillo_bunny", 12),
    ("armadillo_bunny__organizers_16__repeat_04.usd", "ab-org-16", "armadillo_bunny_organizers", "16 organizers", "armadillo_bunny", 16),
    # --- Organizer sweep: mouse limb ---
    ("mouse_limb_11_5_mouse_limb_10_5__uniform_no_cue__repeat_05.usd", "ml-org-nocue", "mouse_limb_organizers", "No cue", "mouse_limb", "no_cue"),
    ("mouse_limb_11_5_mouse_limb_10_5__organizers_03__repeat_04.usd", "ml-org-03", "mouse_limb_organizers", "3 organizers", "mouse_limb", 3),
    ("mouse_limb_11_5_mouse_limb_10_5__organizers_06__repeat_02.usd", "ml-org-06", "mouse_limb_organizers", "6 organizers", "mouse_limb", 6),
    ("mouse_limb_11_5_mouse_limb_10_5__organizers_09__repeat_01.usd", "ml-org-09", "mouse_limb_organizers", "9 organizers", "mouse_limb", 9),
    ("mouse_limb_11_5_mouse_limb_10_5__organizers_12__repeat_01.usd", "ml-org-12", "mouse_limb_organizers", "12 organizers", "mouse_limb", 12),
    ("mouse_limb_11_5_mouse_limb_10_5__organizers_16__repeat_01.usd", "ml-org-16", "mouse_limb_organizers", "16 organizers", "mouse_limb", 16),
]

SECTIONS = [
    ("featured", "Featured reconstructions",
     "Learned trajectories that assemble a source cell population into a target shape."),
    ("armadillo_bunny_organizers", "Organizer sweep · armadillo ↔ bunny",
     "The same reconstruction seeded with an increasing number of organizer cues."),
    ("mouse_limb_organizers", "Organizer sweep · mouse limb",
     "Mouse-limb morphogenesis under an increasing number of organizer cues."),
]


def _point_instancer(stage: Usd.Stage) -> UsdGeom.PointInstancer:
    prim = stage.GetPrimAtPath("/root/cells")
    if prim and prim.GetTypeName() == "PointInstancer":
        return UsdGeom.PointInstancer(prim)
    for p in stage.Traverse():  # fallback: first PointInstancer anywhere
        if p.GetTypeName() == "PointInstancer":
            return UsdGeom.PointInstancer(p)
    raise RuntimeError("no PointInstancer found")


def _proto_sphere_radius(stage: Usd.Stage, pi: UsdGeom.PointInstancer) -> float:
    for target in pi.GetPrototypesRel().GetTargets():
        sp = UsdGeom.Sphere(stage.GetPrimAtPath(target))
        if sp:
            r = sp.GetRadiusAttr().Get()
            if r:
                return float(r)
    return 1.0


def bake_scene(usd_path: Path, meta: dict, out_dir: Path) -> dict:
    stage = Usd.Stage.Open(str(usd_path))
    pi = _point_instancer(stage)
    sphere_r = _proto_sphere_radius(stage, pi)
    pos_attr = pi.GetPositionsAttr()
    scale_attr = pi.GetScalesAttr()
    color_attr = stage.GetPrimAtPath(str(pi.GetPath())).GetAttribute("primvars:displayColor")

    times = sorted(pos_attr.GetTimeSamples()) or [stage.GetStartTimeCode()]
    n_frames = len(times)
    count = len(pos_attr.Get(times[0]))

    # ---- gather all frames -------------------------------------------------
    positions = np.empty((n_frames, count, 3), dtype=np.float32)
    radii = np.empty(n_frames, dtype=np.float32)
    colors_rgb = np.empty((n_frames, count, 3), dtype=np.float32)
    for i, tc in enumerate(times):
        p = np.asarray(pos_attr.Get(tc), dtype=np.float32)
        s = np.asarray(scale_attr.Get(tc), dtype=np.float32)
        positions[i] = p
        radii[i] = float(np.mean(s)) * sphere_r  # isotropic + uniform-within-frame
        c = color_attr.Get(tc)
        if c is None:
            colors_rgb[i] = 0.66
        else:
            c = np.asarray(c, dtype=np.float32)
            colors_rgb[i] = c if len(c) == count else np.broadcast_to(c[:1], (count, 3))

    # ---- quantize positions (uint16 per-scene bbox) ------------------------
    bmin = positions.reshape(-1, 3).min(0)
    bmax = positions.reshape(-1, 3).max(0)
    span = np.where((bmax - bmin) > 1e-9, bmax - bmin, 1.0)
    q = np.round((positions - bmin) / span * 65535.0).astype(np.uint16)
    (out_dir / "positions.u16").write_bytes(q.tobytes())

    # ---- palette + per-cell colour indices (uint8) -------------------------
    flat = np.round(colors_rgb.reshape(-1, 3) * 255.0).astype(np.uint8)
    palette, inverse = np.unique(flat, axis=0, return_inverse=True)
    if len(palette) > 256:  # keep to a byte index; collapse near-duplicates
        step = int(np.ceil(len(palette) / 256))
        keep = palette[::step]
        # remap each colour to the nearest kept swatch
        idx = np.argmin(((flat[:, None, :].astype(int) - keep[None, :, :].astype(int)) ** 2).sum(-1), axis=1)
        palette, inverse = keep, idx
    color_idx = inverse.astype(np.uint8).reshape(n_frames, count)
    (out_dir / "colors.u8").write_bytes(color_idx.tobytes())

    # ---- poster (late frame) ----------------------------------------------
    _render_poster(positions[int(n_frames * 0.9)], colors_rgb[int(n_frames * 0.9)],
                   float(radii[int(n_frames * 0.9)]), out_dir / "poster.png")

    manifest = {
        "name": meta["slug"],
        "label": meta["label"],
        "family": meta["family"],
        "organizer": meta["organizer"],
        "count": int(count),
        "nFrames": int(n_frames),
        "fps": float(stage.GetTimeCodesPerSecond() or 10.0),
        "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
        "bbox": {"min": bmin.tolist(), "max": bmax.tolist()},
        "radius": radii.tolist(),
        "palette": palette.tolist(),
        "files": {"positions": "positions.u16", "colors": "colors.u8"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    size = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"  {meta['slug']:<16} {n_frames:>4}f  {count:>5} cells  "
          f"palette={len(palette):>3}  {size/1e6:5.2f} MB")
    return {
        "slug": meta["slug"], "label": meta["label"], "section": meta["section"],
        "family": meta["family"], "organizer": meta["organizer"],
        "nFrames": int(n_frames), "count": int(count),
        "poster": f"{meta['slug']}/poster.png",
        "dir": meta["slug"],
    }


def _render_poster(pos: np.ndarray, rgb: np.ndarray, radius: float, path: Path) -> None:
    fig = plt.figure(figsize=(4, 4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    # Y-up in USD; matplotlib is Z-up → map (x, y, z)_usd -> (x, z, y) so "up" looks up.
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=np.clip(rgb, 0, 1),
               s=max(2.0, radius * 22.0) ** 2 * 0.25, edgecolors="none", depthshade=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    rng = np.ptp(pos, axis=0).max() * 0.5 or 1.0
    mid = pos.mean(0)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[2] - rng, mid[2] + rng)
    ax.set_zlim(mid[1] - rng, mid[1] + rng)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main() -> None:
    if not USD_ZIP.exists():
        raise SystemExit(f"missing {USD_ZIP} (git-ignored build input — place it in docs/)")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Baking {len(SCENES)} scenes -> {OUT_ROOT}")
    entries = []
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(USD_ZIP) as zf:
        names = set(zf.namelist())
        for fname, slug, section, label, family, organizer in SCENES:
            if fname not in names:
                raise SystemExit(f"{fname} not in {USD_ZIP.name}")
            zf.extract(fname, tmp)
            out_dir = OUT_ROOT / slug
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            entries.append(bake_scene(
                Path(tmp) / fname,
                {"slug": slug, "section": section, "label": label,
                 "family": family, "organizer": organizer},
                out_dir,
            ))
    index = {
        "sections": [
            {"id": sid, "title": title, "blurb": blurb,
             "scenes": [e for e in entries if e["section"] == sid]}
            for sid, title, blurb in SECTIONS
        ]
    }
    (OUT_ROOT / "index.json").write_text(json.dumps(index, indent=None))
    total = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file())
    print(f"Wrote index.json · total gallery assets {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
