#!/bin/bash
COMPLETED=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
ERRORS=$(find checkpoints/ -name "*_ERROR.txt" 2>/dev/null | wc -l)

if squeue -u $USER | grep -q "BeliefMerging"; then
    echo "Status: 🟢 RUNNING | Completed: $COMPLETED | Errors: $ERRORS"
else
    echo "Status: 🔴 NOT RUNNING | Completed: $COMPLETED | Errors: $ERRORS"
fi
