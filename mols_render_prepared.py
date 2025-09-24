#!/usr/bin/env python
# render_prepared_safe.py
# Render .mol/.sdf from prepared/ into PNGs with PyMOL (with guardrails)

import argparse, sys
from pathlib import Path

def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

try:
    from pymol import cmd, preset
except Exception as e:
    die(f"Failed to import PyMOL: {e}\nTip: run in the env that has pymol-open-source.")

def style_object(obj="MOL"):
    preset.ball_and_stick(selection=obj)
    cmd.hide("everything", obj)
    cmd.show("sticks", obj)
    cmd.show("spheres", obj)

    cmd.set("bg_rgb", [1, 1, 1])
    cmd.set("antialias", 3)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", 0.1)
    cmd.set("ambient", 0.5)
    cmd.set("spec_reflect", 0)
    cmd.set("specular", 0.15)
    cmd.set("stick_radius", 0.18)
    cmd.set("sphere_scale", 0.25)
    cmd.set("depth_cue", 0)

    cmd.color("gray80", "elem C")
    cmd.color("red", "elem O")
    cmd.color("slate", "elem N")
    cmd.color("gray98", "elem H")
    cmd.color("tv_orange", "elem S")
    cmd.color("purple", "elem P")

def render_one(in_path: Path, out_png: Path, size=1000, dpi=300, zoom_margin=1.1, transparent=False, overwrite=False):
    if out_png.exists() and not overwrite:
        print(f"[SKIP] Exists: {out_png.name}")
        return True
    try:
        cmd.reinitialize()
        cmd.load(str(in_path), "MOL")
        style_object("MOL")
        if transparent:
            cmd.set("ray_opaque_background", 0)
        cmd.orient("MOL")
        cmd.zoom("MOL", zoom_margin)
        cmd.ray(size, size)
        cmd.png(str(out_png), dpi=dpi)
        print(f"[OK]   Saved: {out_png.name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed render {in_path.name}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Render prepared MOL/SDF into PNGs using PyMOL.")
    ap.add_argument("--root", default="/Users/svlg/MasterThesis/v03_geom/aa_experiments/",
                    help="Root folder that contains experiment dirs (default: your path)")
    ap.add_argument("--experiment", default="0923_2305_al_dipole_energy",
                    help="Experiment name (default matches your example)")
    ap.add_argument("--seed", type=int, default=1, help="Seed suffix in folder name (default: 1)")
    ap.add_argument("--size", type=int, default=1200, help="Image size (square), default 1000")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI, default 300")
    ap.add_argument("--transparent", action="store_true", help="Transparent background")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    args = ap.parse_args()

    base = Path(args.root).expanduser().resolve() / f"{args.experiment}_{args.seed}"
    prepared_dir = base / "prepared"
    renders_dir  = base / "renders"
    if not prepared_dir.is_dir():
        die(f"Prepared folder not found: {prepared_dir}")
    renders_dir.mkdir(parents=True, exist_ok=True)

    mols = sorted([p for p in prepared_dir.iterdir() if p.suffix.lower() in (".mol", ".sdf")])
    if not mols:
        die(f"No .mol/.sdf found in {prepared_dir}")

    ok = 0
    for m in mols:
        out_png = renders_dir / (m.stem + ".png")
        if render_one(m, out_png, size=args.size, dpi=args.dpi,
                      transparent=args.transparent, overwrite=args.overwrite):
            ok += 1

    cmd.quit()
    print(f"[DONE] Rendered {ok}/{len(mols)} molecules to {renders_dir}")

if __name__ == "__main__":
    main()
