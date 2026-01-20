#!/usr/bin/env bash


export PYTHONPATH="/notebooks/CSIRO:${PYTHONPATH:-}"
set -e

# ---- Paths (edit if you really need to) ----
SETUPS_DIR="/notebooks/setups"
VENV_DIR="/notebooks/venvs/pt27cu118"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export COMET_DISABLE_AUTO_LOGGING=1
#export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TRITON_PRINT_AUTOTUNING=1

export DEFAULT_DATA_ROOT="/notebooks/kaggle/csiro"
export DINO_B_WEIGHTS_PATH="/notebooks/kaggle/csiro/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
export DINO_L_WEIGHTS_PATH="/notebooks/kaggle/csiro/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


# 4) Jupyter kernel + handy libs
pip install -U timm ipykernel ipywidgets==8.1.2 comet-ml datasets ruamel.yaml kagglehub
python -m ipykernel install --user --name pt27cu118 --display-name "PyTorch 2.7.1 (cu118)"

if [ -n "${COMET_API_KEY:-}" ]; then
  echo "" >> "$VENV_DIR/bin/activate"
  echo "# Comet ML API key" >> "$VENV_DIR/bin/activate"
  echo "export COMET_API_KEY=\"$COMET_API_KEY\"" >> "$VENV_DIR/bin/activate"
fi
echo "Done. Activate with:  source \"$VENV_DIR/bin/activate\""


case "${1:-}" in
  run)
    cfg="${2:?Usage: $0 run <config>}"
    if [[ "$cfg" != /* ]]; then
      cfg="$SETUPS_DIR/configs/experiments/$cfg"
    fi
    python /notebooks/CSIRO/csiro/scripts/train_cv_func.py --overrides "$cfg"
    ;;
  vit)
    python -c "import timm" >/dev/null 2>&1 || python -m pip install -U timm
    python "$SETUPS_DIR/train_vit.py"
    ;;
  *)
    echo "Usage:"
    echo "  $0 run <config>"
    echo "  $0 vit"
    exit 1
    ;;
esac
