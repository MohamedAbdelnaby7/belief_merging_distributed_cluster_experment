#!/bin/bash
echo "Belief Merging Experiment Monitor"
echo "================================="

# Check SLURM job status
echo "SLURM Job Status:"
if squeue -u $USER | grep -q "BeliefMerging"; then
    echo "  🟢 RUNNING"
    squeue -u $USER | grep "BeliefMerging" | head -5
else
    echo "  🔴 NOT RUNNING"
fi

echo ""

# Count progress
COMPLETED=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
ERRORS=$(find checkpoints/ -name "*_ERROR.txt" 2>/dev/null | wc -l)

echo "Task Progress:"
echo "  ✅ Completed: $COMPLETED"
echo "  ❌ Errors: $ERRORS"

# Estimate total tasks
if [[ -f "configs/${EXPERIMENT_TYPE}_config.json" ]]; then
    TOTAL_TASKS=$(python3 -c "
import json
with open('configs/${EXPERIMENT_TYPE}_config.json', 'r') as f:
    config = json.load(f)
total = len(config['merge_intervals']) * len(config['target_patterns']) * config['n_trials']
print(total)
" 2>/dev/null || echo "unknown")
    
    if [[ "$TOTAL_TASKS" != "unknown" ]]; then
        PROGRESS=$(echo "scale=1; $COMPLETED*100/$TOTAL_TASKS" | bc 2>/dev/null || echo "0")
        echo "  📊 Progress: ${PROGRESS}%"
    fi
fi

echo ""

# Show recent log activity
echo "Recent Activity:"
tail -n 3 logs/experiment_*.out 2>/dev/null | sed 's/^/  /' || echo "  No log files yet"

echo ""
echo "Commands:"
echo "  📊 Full status: squeue -u \$USER"
echo "  📝 Live logs: tail -f logs/experiment_*.out"
echo "  🚫 Cancel job: scancel <job_id>"
