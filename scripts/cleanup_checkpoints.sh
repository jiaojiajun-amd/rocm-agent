#!/bin/bash
# Delete checkpoints with global_step > 30

CHECKPOINT_DIR="./checkpoints"
MAX_STEP=30

echo "Finding checkpoints with step > $MAX_STEP in $CHECKPOINT_DIR..."

find "$CHECKPOINT_DIR" -type d -name "global_step_*" | while read -r dir; do
    step=$(basename "$dir" | sed 's/global_step_//')
    if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -gt "$MAX_STEP" ]; then
        echo "Deleting: $dir (step=$step)"
        rm -rf "$dir"
    fi
done

echo "Done."
