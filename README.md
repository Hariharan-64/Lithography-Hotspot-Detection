Lithography Hotspot Detection - Multi-Representation Learning

Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r Source_Code/requirements.txt

# Run
cd Source_Code
bash run_complete_pipeline.sh
```

## Dataset
- ICCAD-12 benchmark (5 datasets, 164k images)
- Download from: http://www.ispd.cc/contests/12/](https://drive.google.com/file/d/1jx7gDR92sqoIw2Nh4NGwwZwC-qNp2osd/view)
- Extract to: `data/iccad-official/`

## Results
- Individual representations: 49.9% - 96.9% balanced accuracy
- Edge representation best (96.9% on ICCAD-2)
- Fusion degrades performance (-30% on some benchmarks)
- Key finding: No universal best approach

## Files
- `Source_Code/`: Training scripts
- `EXECUTION_INSTRUCTIONS_SIMPLE.md`: Detailed run guide
- `DATASET_PREPROCESSING_DESCRIPTION.md`: Data details

## Issues?
- Memory error → `config.BATCH_SIZE = 8`
- GPU error → `config.DEVICE = "cpu"`
- Dataset not found → Update `DATA_ROOT` in config.py

Duration: 2-3 hours (GPU) / 4-5 hours (CPU)
