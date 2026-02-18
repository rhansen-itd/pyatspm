"""
Run Ingestion Script

Ingests data for a specific intersection folder.
Reads metadata.json to populate the 'metadata' table and configure ingestion.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.atspm.data import (
    init_db,
    import_config,
    run_ingestion,
    run_cycle_processing,
    DatabaseManager
)

INTERSECTIONS_DIR = PROJECT_ROOT / "intersections"

def process_ingestion(target_folder: str, incremental: bool = True):
    folder_path = INTERSECTIONS_DIR / target_folder
    
    if not folder_path.exists():
        print(f"❌ Error: Folder not found: {folder_path}")
        return

    # Load Metadata
    meta_path = folder_path / "metadata.json"
    if not meta_path.exists():
        print(f"❌ Error: metadata.json not found in {target_folder}")
        return
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    print(f"=== Processing {meta.get('intersection_name')} (ID: {meta.get('intersection_id')}) ===")
    
    # Paths
    db_path = folder_path / meta['db_filename']
    config_path = folder_path / "int_cfg.csv"
    raw_dir = folder_path / "raw_data"
    
    # 1. Initialize Database & Sync Metadata
    # This pushes the JSON values into the SQLite 'metadata' table
    print(f"   🔧 Syncing Metadata to DB: {db_path.name}")
    init_db(db_path)
    
    with DatabaseManager(db_path) as manager:
        manager.set_metadata(
            intersection_id=meta.get('intersection_id'),
            timezone=meta.get('timezone'),
            controller_ip=meta.get('controller_ip'),
            detection_type=meta.get('detection_type'),
            detection_ip=meta.get('detection_ip'),
            major_road_route=meta.get('major_road_route'),
            major_road_name=meta.get('major_road_name'),
            minor_road_route=meta.get('minor_road_route'),
            minor_road_name=meta.get('minor_road_name'),
            latitude=meta.get('latitude'),
            longitude=meta.get('longitude'),
            agency_id=meta.get('agency_id')
        )

    # 2. Import Config
    if config_path.exists():
        try:
            import_config(config_path, db_path)
        except Exception as e:
            print(f"   ⚠️  Config import warning: {e}")

    # 3. Run Ingestion
    # Note: Ingestion now relies on the timezone we just saved to DB/Metadata
    print(f"   🚀 Running Ingestion...")
    try:
        run_ingestion(
            db_path=db_path,
            data_dir=raw_dir,
            timezone=meta.get('timezone', 'US/Mountain'),
            incremental=incremental
        )
    except Exception as e:
        print(f"   ❌ Ingestion Failed: {e}")
        return

    # 4. Run Cycle Processing
    print(f"   🔄 Running Cycle Processing...")
    try:
        run_cycle_processing(db_path, reprocess=False)
    except Exception as e:
        print(f"   ❌ Cycle Processing Failed: {e}")

    print("\n✅ Processing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Folder name")
    parser.add_argument("--full", action="store_true", help="Run full reprocessing")
    args = parser.parse_args()

    process_ingestion(args.target, incremental=not args.full)