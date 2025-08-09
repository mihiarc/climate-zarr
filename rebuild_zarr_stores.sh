#!/usr/bin/env bash

set -euo pipefail

# Rebuild Zarr stores for pr, tas, tasmax across regions, with build-time
# longitude normalization (-180..180) and regional clipping.
#
# Usage:
#   ./rebuild_zarr_stores.sh [ssp370]
#
# Environment overrides:
#   VARS="pr tas tasmax"               # variables to build
#   REGIONS="conus alaska ..."         # regions to build
#   DATA_BASE="/abs/path/to/data"      # base input directory
#   OUT_BASE="/abs/path/to/outputs"    # base output directory
#   PYTHON="python"                    # python interpreter

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENARIO="${1:-ssp370}"
VARS="${VARS:-pr tas tasmax}"
REGIONS="${REGIONS:-conus alaska hawaii guam puerto_rico}"
DATA_BASE="${DATA_BASE:-${ROOT_DIR}/data}"
OUT_BASE="${OUT_BASE:-${ROOT_DIR}/climate_outputs/zarr}"
PYTHON_BIN="${PYTHON:-python}"

echo "[INFO] Rebuilding Zarr stores"
echo "       scenario: ${SCENARIO}"
echo "       variables: ${VARS}"
echo "       regions: ${REGIONS}"
echo "       data base: ${DATA_BASE}"
echo "       output base: ${OUT_BASE}"

# Ensure Python can import the package from src/
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

for VAR in ${VARS}; do
  SRC_DIR="${DATA_BASE}/${VAR}/${SCENARIO}"
  if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[WARN] Skipping ${VAR}: missing input dir ${SRC_DIR}" >&2
    continue
  fi

  # Verify there are NetCDF files
  if ! compgen -G "${SRC_DIR}/${VAR}_day_*_${SCENARIO}_*.nc" > /dev/null; then
    echo "[WARN] Skipping ${VAR}: no NetCDF files in ${SRC_DIR}" >&2
    continue
  fi

  echo "\n=== ${VAR^^} ==="
  for REGION in ${REGIONS}; do
    OUT_DIR="${OUT_BASE}/${VAR}/${REGION}/${SCENARIO}"
    OUT_PATH="${OUT_DIR}/${REGION}_${SCENARIO}_${VAR}_daily.zarr"
    mkdir -p "${OUT_DIR}"

    echo "[BUILD] ${VAR}/${REGION} -> ${OUT_PATH}"
    # Call the builder module; it will:
    # - collect *.nc from the input directory
    # - clip to region
    # - normalize longitudes to -180..180
    # - compress and write zarr
    "${PYTHON_BIN}" -m climate_zarr.stack_nc_to_zarr \
      "${SRC_DIR}" \
      -o "${OUT_PATH}" \
      --clip "${REGION}" \
      --compression default \
      --compression-level 5

    # Quick lon sanity check (best-effort)
    "${PYTHON_BIN}" - "${OUT_PATH}" "${VAR}" <<'PY' || true
import xarray as xr, sys
from pathlib import Path
zp = Path(sys.argv[1])
var = sys.argv[2]
try:
  ds = xr.open_zarr(zp)
  lon = None
  for name in ('lon','longitude','x'):
    if name in ds.coords:
      lon = ds[name]; break
  if lon is not None:
    print(f"[CHECK] {var}: lon range [{float(lon.min()):.3f}, {float(lon.max()):.3f}]")
  ds.close()
except Exception as e:
  print(f"[CHECK] {var}: could not open {zp} ({e})")
PY
  done
done

echo "\n[SUCCESS] Rebuild complete."

