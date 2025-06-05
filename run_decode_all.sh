#!/bin/bash

BASE_DIR="checkpoints"

for CKPT_DIR in $(find "$BASE_DIR" -type d -mindepth 2); do
    LAST_CKPT=$(ls "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort -V | tail -n 1)

    if [[ -f "$LAST_CKPT" ]]; then
        echo "=============================="
        echo "Running with checkpoint: $LAST_CKPT"
        ./run_decode.sh "$LAST_CKPT"
        echo "=============================="
        echo
    fi
done
