from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = REPO_ROOT / "parkinson_feature_study"
LOCAL_PACKAGES = STUDY_ROOT / ".python_packages"

for path in (BACKEND_ROOT, STUDY_ROOT, LOCAL_PACKAGES):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
