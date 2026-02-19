import sys
from pathlib import Path

# --- 1. SETUP PATHS (MUST BE FIRST) ----------------------------------------
INTERSECTION_FOLDER = "2068_US-95_and_SH-8"
#INTERSECTION_FOLDER = "201_SH-55_and_Banks-Lowman_Rd"

TEST_DATE = "2026-02-17"  # Update this to a date you know has data

def setup_project_paths(intersection_folder_name):
    """
    1. Locates project root (looks for 'src/atspm').
    2. Adds 'src/' to sys.path so 'import atspm' works.
    3. Sets up database and output paths.
    """
    # Start at current directory and walk up until we find 'src/atspm'
    current_path = Path.cwd()
    while not (current_path / "src" / "atspm").exists():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not locate project root (looking for 'src/atspm').")
        current_path = current_path.parent
    
    root = current_path
    
    # CRITICAL: Add src to Python Path so imports work
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path)) # Insert at 0 to prioritize local code
        print(f"Added to path: {src_path}")
    
    # Setup Intersection Paths
    intersection_dir = root / "intersections" / intersection_folder_name
    
    # Auto-find the .db file
    db_candidates = list(intersection_dir.glob("*.db"))
    if db_candidates:
        db_path = db_candidates[0]
    else:
        db_path = intersection_dir / f"{intersection_folder_name.split('_')[0]}_data.db"

    output_dir = intersection_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    return root, db_path, output_dir

# Execute Setup immediately
ROOT_DIR, DB_PATH, OUTPUT_DIR = setup_project_paths(INTERSECTION_FOLDER)

# --- 2. IMPORTS (MUST BE AFTER SETUP) --------------------------------------
# Now that 'src' is in sys.path, these imports will work
from atspm.data.processing import CycleProcessor
from atspm.reports.generators import PlotGenerator

# --- 3. TEST LOGIC ---------------------------------------------------------


def run_test():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    print(f"--- Step 1: Cycle Validation for {DB_PATH.name} ---")
    
    # Initialize processor (this will fix the table schema if needed)
    processor = CycleProcessor(DB_PATH)
    
    # 1. Backfill ring phases (Safe to run repeatedly)
    print("Running backfill_ring_phases()...")
    updated = processor.backfill_ring_phases()
    print(f"Backfilled {updated} rows.")

    # 2. Verify Data for Test Date
    stats = processor.get_cycle_summary_for_date(TEST_DATE)
    if not stats:
        print(f"Date {TEST_DATE} not found in cycles table. Attempting to process...")
        processor.reprocess_date(TEST_DATE)
        stats = processor.get_cycle_summary_for_date(TEST_DATE)
    
    if stats:
        print(f"Cycle Data for {TEST_DATE}: {stats}")
    else:
        print(f"WARNING: No cycles found for {TEST_DATE}. Plots might be empty.")

    print(f"\n--- Step 2: Generating Reports ---")
    gen = PlotGenerator(DB_PATH, OUTPUT_DIR)
    
    try:
        gen.generate_for_date(TEST_DATE)
        print(f"\nSuccess! Check the output folder:")
        print(f"  {OUTPUT_DIR.resolve() / TEST_DATE}")
    except Exception as e:
        print(f"Error generating reports: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()