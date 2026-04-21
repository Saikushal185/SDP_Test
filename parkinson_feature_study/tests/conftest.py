from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"

if str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
