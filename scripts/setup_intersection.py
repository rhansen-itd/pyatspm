"""
Intersection Setup Script

Creates the standardized folder structure and a comprehensive metadata.json
for a new intersection.

Usage:
    python scripts/setup_intersection.py --id 2068 --name "US-95 & SH-8" --tz "US/Pacific"
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

INTERSECTIONS_DIR = PROJECT_ROOT / "intersections"


def sanitize_name(text: str) -> str:
    """Sanitize text for use as a folder name."""
    text = text.replace("&", "and")
    text = text.replace(" ", "_")
    text = re.sub(r'[^\w\-.]', '', text)
    return text


def setup_intersection(int_id: str, int_name: str, timezone: str):
    """Sets up the folder structure and metadata."""

    # 1. Create the standardized folder name
    safe_name = sanitize_name(int_name)
    folder_name = f"{int_id}_{safe_name}"
    target_dir = INTERSECTIONS_DIR / folder_name

    # 2. Define standard paths
    db_path = target_dir / f"{int_id}_data.db"
    config_path = target_dir / "int_cfg.csv"
    raw_dir = target_dir / "raw_data"
    outputs_dir = target_dir / "outputs"
    metadata_path = target_dir / "metadata.json"

    print(f"Creating intersection environment for ID {int_id}...")
    print(f"   📂 Target: {target_dir}")

    # 3. Create Directories
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    # 4. Create Metadata JSON (skip if already exists to avoid clobbering edits)
    if metadata_path.exists():
        print(f"   ⏭️  metadata.json already exists — skipping (delete it to regenerate)")
    else:
        metadata = {
            # --- System Identifiers (Required) ---
            "intersection_id": int_id,
            "intersection_name": int_name,
            "timezone": timezone,
            "folder_name": folder_name,
            "db_filename": db_path.name,

            # --- Operational Fields (Optional - fill in or leave null) ---
            "controller_ip": None,      # e.g., "10.71.10.50"
            "detection_type": None,     # e.g., "Radar", "Loops", "Video"
            "detection_ip": None,       # e.g., "10.71.10.51"
            "agency_id": None,          # e.g., "ITD-D2"

            # --- Geographic Fields (Optional - fill in or leave null) ---
            "major_road_route": None,   # e.g., "US-95"
            "major_road_name": None,    # e.g., "Main St"
            "minor_road_route": None,   # e.g., "SH-8"
            "minor_road_name": None,    # e.g., "Troy Hwy"

            # --- Coordinates (Optional - fill in or leave null) ---
            "latitude": None,           # e.g., 46.732
            "longitude": None           # e.g., -117.001
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"   📄 Created metadata.json (Timezone: {timezone})")

    # 5. Create Placeholder Config (if missing)
    if not config_path.exists():
        print(f"   📄 Creating empty config template: {config_path.name}")
        with open(config_path, 'w') as f:
            f.write("Category,Parameter,Value\n")

    print("\n✅ Setup Complete.")
    print(f"   1. Update Metadata:        {metadata_path.relative_to(PROJECT_ROOT)}")
    print(f"   2. Drop .datZ files into:  {raw_dir.relative_to(PROJECT_ROOT)}")
    print(f"   3. Run ingestion:          python scripts/run_ingestion.py --target \"{folder_name}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Numeric ID")
    parser.add_argument("--name", required=True, help="Descriptive Name")
    parser.add_argument("--tz", default="US/Mountain", help="Timezone")
    args = parser.parse_args()

    setup_intersection(args.id, args.name, args.tz)