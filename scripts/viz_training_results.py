"""CLI wrapper for generating training result visualizations."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.viz import generate_report
from app.ml.registry import _MODELS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization charts from training results"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Model version number or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="forecast",
        help="Model kind (default: forecast)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated images in default viewer (Windows: start, macOS: open, Linux: xdg-open)",
    )
    args = parser.parse_args()

    # Determine model path
    if args.version == "latest":
        candidates = list(_MODELS_DIR.glob(f"{args.kind}_v*.ubj"))
        if not candidates:
            print(f"❌ No saved model of kind '{args.kind}' found in {_MODELS_DIR}")
            sys.exit(1)

        def _version(p: Path) -> int:
            import re

            m = re.search(r"_v(\d+)\.ubj$", p.name)
            return int(m.group(1)) if m else 0

        best = max(candidates, key=_version)
        meta_path = best.with_suffix(".json")
        version_num = _version(best)
    else:
        meta_path = _MODELS_DIR / f"{args.kind}_v{args.version}.json"
        booster_path = _MODELS_DIR / f"{args.kind}_v{args.version}.ubj"
        if not meta_path.exists():
            print(f"❌ Model metadata not found: {meta_path}")
            sys.exit(1)
        if not booster_path.exists():
            print(f"⚠️  Model booster not found (feature importance won't be generated): {booster_path}")

    print(f"📊 Generating visualizations for {args.kind} v{version_num}...")

    # Generate report
    try:
        booster_path = meta_path.with_suffix(".ubj")
        image_paths = generate_report(meta_path, booster_path if booster_path.exists() else None)

        print(f"\n✅ Generated {len(image_paths)} visualization(s):")
        for img_path in image_paths:
            print(f"   • {img_path}")

        # Open images if requested
        if args.open:
            import platform
            import subprocess

            os_name = platform.system()
            for img_path in image_paths:
                try:
                    if os_name == "Windows":
                        subprocess.Popen(["start", str(img_path)], shell=True)
                    elif os_name == "Darwin":
                        subprocess.Popen(["open", str(img_path)])
                    else:  # Linux and others
                        subprocess.Popen(["xdg-open", str(img_path)])
                except Exception as e:
                    print(f"   ⚠️  Could not open {img_path}: {e}")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
