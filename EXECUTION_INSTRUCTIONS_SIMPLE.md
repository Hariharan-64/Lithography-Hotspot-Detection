# Execution Instructions - Quick Guide

## Prerequisites
- Python 3.8+
- 8GB RAM
- 50GB disk space for dataset
- GPU recommended (optional)

---

## Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/Hariharan-64/Lithography-Hotspot-Detection.git
cd lithography-hotspot-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Source_Code/requirements.txt
```

---

## Download Dataset

1. Download `iccad_official.rar` from http://www.ispd.cc/contests/12/
2. Extract to `data/iccad-official/`
3. Verify: `ls data/iccad-official/` should show `iccad1, iccad2, iccad3, iccad4, iccad5`

---

## Run (2-3 hours)

```bash
cd Source_Code

# Run everything at once
bash run_complete_pipeline.sh

# OR run separately:
python3 group1_hotspot_detection.py        # ~30-45 min
python3 group1_hotspot_detection_part2.py  # ~60-90 min
```

**Output:** Results saved in `Results/` folder

---

## Check Results

```bash
# View results
cat Results/results_summary.txt

# Or check JSON files
ls Results/*.json
```

---

## Issues?

| Problem | Solution |
|---------|----------|
| "Module not found" | `pip install -r Source_Code/requirements.txt` |
| "Out of memory" | Edit `Source_Code/config.py`: `BATCH_SIZE = 8` |
| "Dataset not found" | Update `DATA_ROOT` path in `Source_Code/config.py` |
| "No GPU" | Edit `Source_Code/config.py`: `DEVICE = "cpu"` |

---

Results will be in `Results/` folder after execution.
