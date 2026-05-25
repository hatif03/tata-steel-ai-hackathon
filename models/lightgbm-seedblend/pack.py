import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.submission_pack import pack_method

METHOD_DIR = Path(__file__).resolve().parent
pack_method(METHOD_DIR)
print("Packaged lightgbm-seedblend.")
