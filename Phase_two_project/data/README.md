# Data Directory

This directory contains the data files used in the LLM vs Traditional ML comparison project.

## Files

### Raw Data
- `total_data.csv` - Symbolic link to the main dataset from Pre-processed_data/
  - Contains 5,331 samples with 21 features
  - Target variable: Health_status (0=Good Health, 1=Poor Health)
  - Features include physical activity metrics, demographics, and socioeconomic factors

### Processed Data
- `train_data.csv` - Training dataset (70% of total data)
- `test_data.csv` - Test dataset (30% of total data)
- `llm_samples.csv` - Smaller sample for LLM evaluation (200 samples)

## Data Source
The original data comes from NHANES (National Health and Nutrition Examination Survey) and has been pre-processed in the main project to create the `total_data.csv` file.

## Usage
Data files are loaded using the `src.data_preparation.data_loader` module, which handles:
- Loading CSV files
- Data validation
- Type checking
- Missing value detection

## Symbolic Links
To avoid data duplication, this project uses symbolic links to reference the original data files. This ensures:
- Single source of truth for data
- No storage duplication
- Easy updates when source data changes
