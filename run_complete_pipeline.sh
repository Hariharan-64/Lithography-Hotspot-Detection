#!/bin/bash

################################################################################
# GROUP 1 - MULTI-REPRESENTATION LEARNING
# Complete Pipeline Execution Script
# Runs all 15 days of experiments in sequence
################################################################################

set -e  # Exit on any error

OUTPUT_DIR="/mnt/user-data/outputs"
LOG_FILE="${OUTPUT_DIR}/execution_log.txt"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "================================================================================"
echo "GROUP 1 - LITHOGRAPHY HOTSPOT DETECTION: MULTI-REPRESENTATION LEARNING"
echo "================================================================================"
echo ""
echo "Execution Start: $(date)" | tee ${LOG_FILE}
echo ""

# ============================================================================
# PART 1: DATA ANALYSIS & BASELINE
# ============================================================================
echo ""
echo "================================================================================"
echo "PART 1: DATA ANALYSIS & BASELINE MODELS"
echo "================================================================================"
echo ""

python3 /home/claude/group1_hotspot_detection.py 2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Part 1 completed successfully" | tee -a ${LOG_FILE}
else
    echo "✗ Part 1 FAILED" | tee -a ${LOG_FILE}
    exit 1
fi

# ============================================================================
# PART 2: REPRESENTATIONS, FUSION & ABLATION
# ============================================================================
echo ""
echo "================================================================================"
echo "PART 2: REPRESENTATIONS, FUSION MODELS & ABLATION STUDY"
echo "================================================================================"
echo ""

python3 /home/claude/group1_hotspot_detection_part2.py 2>&1 | tee -a ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "✓ Part 2 completed successfully" | tee -a ${LOG_FILE}
else
    echo "✗ Part 2 FAILED" | tee -a ${LOG_FILE}
    exit 1
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "EXECUTION COMPLETE"
echo "================================================================================"
echo ""
echo "Output files generated in: ${OUTPUT_DIR}"
echo ""
ls -lh ${OUTPUT_DIR} | tee -a ${LOG_FILE}
echo ""
echo "Execution End: $(date)" | tee -a ${LOG_FILE}
echo "================================================================================"
