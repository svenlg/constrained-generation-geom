# #!/usr/bin/env python
# # render_prepared_safe.py
# # Render .mol/.sdf into PNGs with PyMOL (with guardrails + single-file mode)

# import argparse, sys
# from pathlib import Path

# def die(msg, code=1):
#     print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

# try:
#     from pymol import cmd, preset
# except Exception as e:
#     die(f"Failed to import PyMOL: {e}\nTip: run in the env that has pymol-open-source.")

# def style_object(obj="MOL"):
#     preset.ball_and_stick(selection=obj)
#     cmd.hide("everything", obj)
#     cmd.show("sticks", obj)
#     cmd.show("spheres", obj)

#     cmd.set("bg_rgb", [1, 1, 1])
#     cmd.set("antialias", 3)
#     cmd.set("ray_trace_mode", 1)
#     cmd.set("ray_trace_gain", 0.1)
#     cmd.set("ambient", 0.5)
#     cmd.set("spec_reflect", 0)
#     cmd.set("specular", 0.15)
#     cmd.set("stick_radius", 0.18)
#     cmd.set("sphere_scale", 0.25)
#     cmd.set("depth_cue", 0)

#     cmd.color("gray80", "elem C")
#     cmd.color("red", "elem O")
#     cmd.color("slate", "elem N")
#     cmd.color("gray98", "elem H")
#     cmd.color("tv_orange", "elem S")
#     cmd.color("purple", "elem P")

# def render_one(
#     in_path: Path,
#     out_png: Path,
#     x_size=1000,
#     y_size=1000,
#     dpi=300,
#     buffer=2.0,
#     transparent=False,
#     overwrite=False,
#     orthoscopic=False,
# ):
#     if out_png.exists() and not overwrite:
#         print(f"[SKIP] Exists: {out_png.name}")
#         return True
#     try:
#         cmd.reinitialize()
#         cmd.load(str(in_path), "MOL")
#         style_object("MOL")

#         if transparent:
#             cmd.set("ray_opaque_background", 0)

#         cmd.set("orthoscopic", 1 if orthoscopic else 0)

#         # Some PyMOL builds don't support near/far clip as settings; don't fail if they do not exist
#         try:
#             cmd.set("near_clip", -5)
#             cmd.set("far_clip", 5)
#         except Exception:
#             # Fallback: aggressively widen the view slab; harmless if already big enough
#             try:
#                 cmd.clip("slab", 10000)
#             except Exception:
#                 pass

#         cmd.orient("MOL")        # align principal axes & center-of-mass
#         cmd.center("MOL")
#         cmd.zoom("MOL", buffer=float(buffer))   # add Å padding around bbox

#         cmd.ray(int(x_size), int(y_size))
#         cmd.png(str(out_png), dpi=int(dpi))
#         print(f"[OK]   Saved: {out_png.name}")
#         return True
#     except Exception as e:
#         print(f"[WARN] Failed render {in_path.name}: {e}")
#         return False


# def main(args):
#     if args.mol:
#         in_path = Path(args.mol).expanduser().resolve()
#         if not in_path.is_file() or in_path.suffix.lower() not in (".mol", ".sdf"):
#             die(f"--mol must be an existing .mol/.sdf file: {in_path}")
#         # Resolve output
#         if args.out:
#             outp = Path(args.out).expanduser().resolve()
#             if outp.is_dir():
#                 out_png = outp / (in_path.stem + ".png")
#                 outp.mkdir(parents=True, exist_ok=True)
#             else:
#                 out_png = outp
#                 out_png.parent.mkdir(parents=True, exist_ok=True)
#         else:
#             out_png = in_path.with_suffix(".png")
#         ok = render_one(
#             in_path, out_png,
#             x_size=args.x_size,
#             y_size=args.y_size,
#             dpi=args.dpi,
#             buffer=args.buffer,
#             transparent=args.transparent,
#             overwrite=args.overwrite,
#             orthoscopic=args.orthoscopic,
#         )
#         cmd.quit()
#         print(f"[DONE] Rendered 1/1 to {out_png.parent}")
#         sys.exit(0 if ok else 1)

#     # Batch mode (original behavior)
#     base = Path(args.root).expanduser().resolve() / f"{args.experiment}_{args.seed}"
#     prepared_dir = base / "prepared"
#     renders_dir  = base / "renders"
#     if not prepared_dir.is_dir():
#         die(f"Prepared folder not found: {prepared_dir}")
#     renders_dir.mkdir(parents=True, exist_ok=True)

#     mols = sorted([p for p in prepared_dir.iterdir() if p.suffix.lower() in (".mol", ".sdf")])
#     if not mols:
#         die(f"No .mol/.sdf found in {prepared_dir}")

#     ok = 0
#     for m in mols:
#         out_png = renders_dir / (m.stem + ".png")
#         if render_one(
#             m, out_png,
#             x_size=args.x_size,
#             y_size=args.y_size,
#             dpi=args.dpi,
#             buffer=args.buffer,
#             transparent=args.transparent,
#             overwrite=args.overwrite,
#             orthoscopic=args.orthoscopic,
#         ):
#             ok += 1

#     cmd.quit()
#     print(f"[DONE] Rendered {ok}/{len(mols)} molecules to {renders_dir}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="Render MOL/SDF into PNGs using PyMOL.")
#     # Batch-mode (original)
#     ap.add_argument("--root", default="/Users/svlg/MasterThesis/v03_geom/aa_experiments/",
#                     help="Root folder that contains experiment dirs")
#     ap.add_argument("--experiment", default="0923_2305_al_dipole_energy",
#                     help="Experiment name (folder prefix)")
#     ap.add_argument("--seed", type=int, default=1, help="Seed suffix in folder name")
#     # Single-file mode
#     ap.add_argument("--mol", type=str, help="Path to a single .mol/.sdf to render (overrides --root/--experiment/--seed)")
#     ap.add_argument("--out", type=str, help="Output PNG path (file or directory). If directory, file name = <mol>.png")
#     # Rendering params
#     ap.add_argument("--x_size", type=int, default=1000, help="Image width (pixels)")
#     ap.add_argument("--y_size", type=int, default=700, help="Image height (pixels)")
#     # Add these args in main()
#     ap.add_argument("--auto-aspect", action="store_true",
#                 help="Auto-set viewport to molecule XY aspect ratio to reduce whitespace")
#     ap.add_argument("--dpi", type=int, default=300, help="PNG DPI metadata")
#     ap.add_argument("--buffer", type=float, default=2.5, help="Zoom buffer in Å around molecule bbox (prevents cropping)")
#     ap.add_argument("--orthoscopic", action="store_true", help="Use orthographic projection (no perspective)")
#     ap.add_argument("--transparent", action="store_true", help="Transparent background")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
#     args = ap.parse_args()
#     main(args)

# # python mols_single_plot.py --mol /Users/svlg/MasterThesis/v03_geom/aa_experiments/0923_2305_al_dipole_energy_1/prepared/iter_000060_idx_004.mol --buffer 2 --overwrite
# # python mols_single_plot.py --mol /Users/svlg/MasterThesis/v03_geom/aa_experiments/0923_2305_al_dipole_energy_1/prepared/iter_000060_idx_011.mol --buffer 0.3 --overwrite
# # python mols_single_plot.py --mol /Users/svlg/MasterThesis/v03_geom/aa_experiments/0923_2305_al_dipole_energy_1/prepared/iter_000060_idx_037.mol --buffer 0.3 --overwrite
# # python mols_single_plot.py --mol /Users/svlg/MasterThesis/v03_geom/aa_experiments/0923_2305_al_dipole_energy_1/prepared/iter_000060_idx_055.mol --buffer 0.3 --overwrite

#!/usr/bin/env python
# render_prepared_safe.py
# Render .mol/.sdf into PNGs with PyMOL (with guardrails + single-file & iter/idx modes)

import argparse, sys, glob
from pathlib import Path

def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

try:
    from pymol import cmd, preset
except Exception as e:
    die(f"Failed to import PyMOL: {e}\nTip: run in the env that has pymol-open-source.")

# ---------------- style ----------------
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

# ---------------- helpers ----------------
def iter_tag_str(iter_id: int) -> str:
    """Format iteration folder name suffix consistent with prepare_mols_all.py (iterXX for <100)."""
    if iter_id < 0:
        return "NA"
    return f"{iter_id:02d}" if iter_id < 100 else f"{iter_id}"

def auto_aspect_resize(y_size: int, x_size: int) -> tuple[int, int]:
    """
    Use molecule XY bounding box to set x_size keeping y_size fixed.
    Called after loading & orienting.
    """
    try:
        (xmin, ymin, _zmin), (xmax, ymax, _zmax) = cmd.get_extent("MOL")
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        aspect = dx / dy  # width/height
        new_x = max(int(round(y_size * aspect)), 64)
        return y_size, new_x if new_x > 0 else x_size
    except Exception:
        return y_size, x_size

# ---------------- core ----------------
def render_one(
    in_path: Path,
    out_png: Path,
    x_size=1000,
    y_size=1000,
    dpi=300,
    buffer=2.0,
    transparent=False,
    overwrite=False,
    orthoscopic=False,
    auto_aspect=False,
):
    if out_png.exists() and not overwrite:
        print(f"[SKIP] Exists: {out_png.name}")
        return True
    try:
        cmd.reinitialize()
        cmd.load(str(in_path), "MOL")
        style_object("MOL")

        if transparent:
            cmd.set("ray_opaque_background", 0)

        cmd.set("orthoscopic", 1 if orthoscopic else 0)

        # Best-effort wide clipping
        try:
            cmd.set("near_clip", -5)
            cmd.set("far_clip", 5)
        except Exception:
            try:
                cmd.clip("slab", 10000)
            except Exception:
                pass

        cmd.orient("MOL")
        cmd.center("MOL")
        cmd.zoom("MOL", buffer=float(buffer))

        # Optional viewport aspect fit
        if auto_aspect:
            y_size, x_size = auto_aspect_resize(y_size, x_size)

        cmd.ray(int(x_size), int(y_size))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cmd.png(str(out_png), dpi=int(dpi))
        print(f"[OK]   Saved: {out_png}")
        return True
    except Exception as e:
        print(f"[WARN] Failed render {in_path.name}: {e}")
        return False

def find_mol_in_iter(prepared_root: Path, iter_id: int, idx: int) -> Path | None:
    """
    Locate a prepared file in prepared/iterXX/ matching *_idx_{idx:03d}.(mol|sdf).
    Prefer .mol if both exist.
    """
    it_dir = prepared_root / f"iter{iter_tag_str(iter_id)}"
    if not it_dir.is_dir():
        return None
    patt = f"*__idx_{idx:03d}.*"  # double underscore in case stems have underscores; we'll glob both
    # Be flexible: many files look like <bin-stem>_idx_000.mol
    candidates = list(it_dir.glob(f"*_*idx_{idx:03d}.mol")) + \
                 list(it_dir.glob(f"*_*idx_{idx:03d}.sdf")) + \
                 list(it_dir.glob(f"*idx_{idx:03d}.mol")) + \
                 list(it_dir.glob(f"*idx_{idx:03d}.sdf"))
    if not candidates:
        # last-resort broad glob
        candidates = [Path(p) for p in glob.glob(str(it_dir / f"*idx_{idx:03d}.*"))]
    if not candidates:
        return None
    # Prefer .mol
    candidates.sort(key=lambda p: (p.suffix.lower() != ".mol", p.name))
    return candidates[0]

def collect_all_prepared(prepared_root: Path) -> list[Path]:
    """
    Gather all .mol/.sdf under prepared/, including per-iteration folders.
    """
    mols = []
    # new layout
    for it_dir in sorted(prepared_root.glob("iter*")):
        mols.extend(sorted([p for p in it_dir.iterdir() if p.suffix.lower() in (".mol", ".sdf")]))
    # backwards-compat: flat files directly under prepared/
    mols.extend(sorted([p for p in prepared_root.iterdir() if p.suffix.lower() in (".mol", ".sdf")]))
    return mols

# ---------------- main ----------------
def main(args):
    # Mode A: explicit file path
    if args.mol:
        in_path = Path(args.mol).expanduser().resolve()
        if not in_path.is_file() or in_path.suffix.lower() not in (".mol", ".sdf"):
            die(f"--mol must be an existing .mol/.sdf file: {in_path}")

        # Resolve output
        if args.out:
            outp = Path(args.out).expanduser().resolve()
            if outp.is_dir():
                outp.mkdir(parents=True, exist_ok=True)
                out_png = outp / (in_path.stem + ".png")
            else:
                out_png = outp
                out_png.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_png = in_path.with_suffix(".png")

        ok = render_one(
            in_path, out_png,
            x_size=args.x_size, y_size=args.y_size, dpi=args.dpi, buffer=args.buffer,
            transparent=args.transparent, overwrite=args.overwrite,
            orthoscopic=args.orthoscopic, auto_aspect=args.auto_aspect,
        )
        cmd.quit()
        print(f"[DONE] Rendered 1/1 to {out_png.parent}")
        sys.exit(0 if ok else 1)

    # Common roots
    base = Path(args.root).expanduser().resolve() / f"{args.experiment}_{args.seed}"
    prepared_root = base / "prepared"
    renders_root  = base / "renders"
    if not prepared_root.is_dir():
        die(f"Prepared folder not found: {prepared_root}")

    # Mode B: iter+idx -> single-target render into renders/iterXX/idx.png
    if args.iter is not None and args.idx is not None:
        it_tag = iter_tag_str(args.iter)
        in_path = find_mol_in_iter(prepared_root, args.iter, args.idx)
        if in_path is None:
            die(f"Could not find prepared file for iter={args.iter} idx={args.idx:03d} in {prepared_root}/iter{it_tag}")
        out_dir = renders_root / f"iter{it_tag}"
        out_png = out_dir / f"{args.idx:03d}.png"
        ok = render_one(
            in_path, out_png,
            x_size=args.x_size, y_size=args.y_size, dpi=args.dpi, buffer=args.buffer,
            transparent=args.transparent, overwrite=args.overwrite,
            orthoscopic=args.orthoscopic, auto_aspect=args.auto_aspect,
        )
        cmd.quit()
        print(f"[DONE] Rendered 1/1 to {out_dir}")
        sys.exit(0 if ok else 1)

    # Mode C: iter-only -> batch that iteration to renders/iterXX/
    if args.iter is not None and args.idx is None:
        it_tag = iter_tag_str(args.iter)
        it_dir = prepared_root / f"iter{it_tag}"
        if not it_dir.is_dir():
            die(f"Iteration folder not found: {it_dir}")
        out_dir = renders_root / f"iter{it_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        mols = sorted([p for p in it_dir.iterdir() if p.suffix.lower() in (".mol", ".sdf")])
        if not mols:
            die(f"No .mol/.sdf found in {it_dir}")

        ok = 0
        for m in mols:
            # try to extract idx
            stem = m.stem
            idx_str = "unknown"
            for token in stem.split("_"):
                if token.isdigit() and len(token) == 3:
                    idx_str = token
            out_png = out_dir / (f"{idx_str}.png" if idx_str != "unknown" else (stem + ".png"))
            if render_one(
                m, out_png,
                x_size=args.x_size, y_size=args.y_size, dpi=args.dpi, buffer=args.buffer,
                transparent=args.transparent, overwrite=args.overwrite,
                orthoscopic=args.orthoscopic, auto_aspect=args.auto_aspect,
            ):
                ok += 1

        cmd.quit()
        print(f"[DONE] Rendered {ok}/{len(mols)} molecules to {out_dir}")
        sys.exit(0 if ok == len(mols) else 1)

    # Mode D: full-batch over all prepared files (new per-iter layout + flat fallback)
    renders_root.mkdir(parents=True, exist_ok=True)
    mols = collect_all_prepared(prepared_root)
    if not mols:
        die(f"No .mol/.sdf found under {prepared_root}")

    ok = 0
    for m in mols:
        # route per-iter into renders/iterXX/
        parent = m.parent
        if parent.name.startswith("iter"):
            out_dir = renders_root / parent.name
        else:
            out_dir = renders_root
        out_dir.mkdir(parents=True, exist_ok=True)

        # prefer nice idx-based filename if present
        stem = m.stem
        idx_str = None
        parts = stem.split("_")
        for k in range(len(parts)-1):
            if parts[k] == "idx" and (k+1) < len(parts):
                cand = parts[k+1]
                if len(cand) == 3 and cand.isdigit():
                    idx_str = cand
                    break
        out_png = out_dir / (f"{idx_str}.png" if idx_str else (stem + ".png"))

        if render_one(
            m, out_png,
            x_size=args.x_size, y_size=args.y_size, dpi=args.dpi, buffer=args.buffer,
            transparent=args.transparent, overwrite=args.overwrite,
            orthoscopic=args.orthoscopic, auto_aspect=args.auto_aspect,
        ):
            ok += 1

    cmd.quit()
    print(f"[DONE] Rendered {ok}/{len(mols)} molecules to {renders_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render MOL/SDF into PNGs using PyMOL.")
    # Batch-mode roots
    ap.add_argument("--root", default="/Users/svlg/MasterThesis/v03_geom/aa_experiments/",
                    help="Root folder that contains experiment dirs")
    ap.add_argument("--experiment", default="0923_2305_al_dipole_energy",
                    help="Experiment name (folder prefix)")
    ap.add_argument("--seed", type=int, default=1, help="Seed suffix in folder name")
    # Single-file (direct path)
    ap.add_argument("--mol", type=str,
                    help="Path to a single .mol/.sdf to render (overrides --root/--experiment/--seed)")
    ap.add_argument("--out", type=str,
                    help="Output PNG path (file or directory). If directory, file name = <mol>.png")
    # New: iteration/index addressing (uses prepared/iterXX/*_idx_XXX.*)
    ap.add_argument("--iter", type=int, default=60, help="Iteration number to read from prepared/iterXX/")
    ap.add_argument("--idx", type=int, help="Molecule index (three-digit) inside that iteration")
    # Rendering params
    ap.add_argument("--x_size", type=int, default=1000, help="Image width (pixels)")
    ap.add_argument("--y_size", type=int, default=800, help="Image height (pixels)")
    ap.add_argument("--auto-aspect", action="store_true",
                    help="Auto-set viewport to molecule XY aspect ratio to reduce whitespace")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI metadata")
    ap.add_argument("--buffer", type=float, default=1.5,
                    help="Zoom buffer in Å around molecule bbox (prevents cropping)")
    ap.add_argument("--orthoscopic", action="store_true", help="Use orthographic projection (no perspective)")
    ap.add_argument("--transparent", action="store_true", help="Transparent background")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    args = ap.parse_args()
    main(args)
