#!/usr/bin/env zsh
set -u
# Test script: activate conda env, run import checks, instantiate detectors, run demo.py and collect logs

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/demo_run.log"

echo "Test started: $(date)" | tee "$LOG_FILE"

echo "Activating conda env 'depth-clean'..." | tee -a "$LOG_FILE"
eval "$($HOME/miniforge3/bin/conda shell.zsh hook)" 2>&1 | tee -a "$LOG_FILE"
conda activate depth-clean 2>&1 | tee -a "$LOG_FILE"

echo "--- Python module import checks ---" | tee -a "$LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import importlib, traceback
modules = ["numpy","torch","open3d","tensorflow","cv2"]
for m in modules:
    try:
        importlib.import_module(m)
        print('[OK]', m)
    except Exception:
        print('[ERR]', m)
        traceback.print_exc()
print('\nPython executable:', __import__('sys').executable)
print('Platform:', __import__('platform').platform())
PY

echo "--- Instantiate ObjectDetectorAPI and HitNetEstimator ---" | tee -a "$LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import traceback
try:
    from object_detector import ObjectDetectorAPI
    print('ObjectDetectorAPI import OK')
    od = ObjectDetectorAPI()
    print('ObjectDetectorAPI.ready =', getattr(od, 'ready', None))
except Exception:
    print('ObjectDetectorAPI failed:')
    traceback.print_exc()

try:
    from disparity_estimator.hitnet_disparity_estimator import HitNetEstimator
    print('HitNetEstimator import OK')
    hn = HitNetEstimator()
    print('HitNetEstimator instantiated')
except Exception:
    print('HitNetEstimator failed:')
    traceback.print_exc()
PY

echo "--- Running demo.py (this may take a while) ---" | tee -a "$LOG_FILE"
python "$REPO_ROOT/demo.py" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]:-0}

echo "Demo finished with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
echo "Test finished: $(date)" | tee -a "$LOG_FILE"

exit $EXIT_CODE
