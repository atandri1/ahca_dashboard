# AHCA Regulatory Citations Dashboard

Streamlit dashboard for exploring regulatory citation data.

## Quickstart

1. Install dependencies (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

2. Add your data (choose one):

   Raw Excel inputs:

   - Put the regional `*.xlsx` files in `data/raw/`
   - Build the processed dataset:

   ```bash
   python scripts/build_analysis_dataset.py --input data/raw --output data/processed
   ```

   Prebuilt processed CSV:

   - Put your dataset at `data/processed/analysis_dataset_jkl_top10tags.csv` (or point the app to a different CSV in the sidebar)

3. Run the dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

## Input Data

### Raw Excel Files

The build script expects a folder of regional `*.xlsx` files and infers the CMS region from the filename:

- Examples: `*_reg1.xlsx`, `*_reg10.xlsx`, `*_reg5a.xlsx`, `*_reg5b.xlsx`
- Note: `reg5a` and `reg5b` are treated as Region 5

Running the build script creates:

- `data/processed/analysis_dataset_jkl_top10tags.csv`
- `data/processed/regional_summary_statistics.csv`
- `data/processed/tag_summary_statistics.csv`

### Processed CSV

If you already have a processed dataset, place it at `data/processed/analysis_dataset_jkl_top10tags.csv`. The app can also
load a CSV from any path you provide in the sidebar.

