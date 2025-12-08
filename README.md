# Large scale livestock workers and mental health in the US 

## Purpose
This project combines US health, crime, population and agriculture data into one large, unbalanced panel. 
Data extends from 1999-2022 with different ranges depending on source year. 
The purpose of this project is to develop a comprehensive, repeated cross-section of data. 
We are interested in identifying causal relationships between large scale livestock, particularly slaughterhouse work, and mental health among US agriculture workers. 
## Requirements

- Python 3.x  
- Packages listed in `packages.py` (manually recorded below):

channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.11
  - pandas
  - numpy
  - scikit-learn
  - pyarrow
  - fastparquet
  - matplotlib
  - seaborn
  - jupyter
  - pip
  - pip:
      - pyreadstat
- Useful functions, particularly for standardizing geographic unit in `functions.py`
- Raw data in shared Dropbox with the following structure:

## Raw data layout (local, not in repo)

Raw inputs live in a separate `Data/` directory (not tracked in this GitHub repo), with this structure:

Data/

    └── raw/
      ├── cdc/              # CDC / health outcome files
      ├── crime/            # crime and social-outcome files
      ├── fips/             # FIPS / geographic crosswalks
      ├── mental/           # mental-health specific inputs
      ├── population/       # population and demographics
      └── usda/             # USDA / slaughterhouse / ag data

The Python scripts `script0a`–`script0d` read one or more files from each of these folders.  
To see exactly which input filenames are required, search the scripts for `read_csv`, `read_parquet`, etc.

## Usage

Run scripts in sequence:

### Data cleaning and variable generation
1. `packages.py`, `functions.py`
2. `script0a-...` through `script0d-...` (raw ingestion, for data cleaning the files)
   --> end up with interim dataframes for each data source in folder ~Mental/Data/clean
4. `script1-review-main-data.py`  
5. `script2-aggregate-merge.py`
    --> end up with large dataframe of universe of variables, affixed to unique FIPS<>YEAR ~Mental/Data/merged
### Analysis 
6. `script3-ridge.py`  (in progress)

Intermediate datasets for testing and review will be created in local dropbox `interim-dfs-copy/`.

---

# Input Path Audit
### Below each script is the dataset required for script execution

#### `script0a-pop-fips-raw-merge.py`
**Input reads found:**

1. `pd.read_csv("data/raw/population.csv")`  
2. `pd.read_csv("data/raw/fips_crosswalk.csv")`
**Path type:**  
Relative → **portable**.  
These files must live in `data/raw/`.


#### `script0b-ag-raw.py`
**Input reads found:**

1. `pd.read_dta` + `\usda.dta`
2. Note: This will be updated to pull data directly from USDA API, as opposed to hard-coded .dta files 
**Path type:**  
Relative → **portable**.


#### `script0c-health-raw.py`
**Input reads found:**

1. `pd.read_csv("data/raw/cdc_health_measures.csv")`  
2. `pd.read_csv("data/raw/county_health_rankings.csv")`
**Path type:**  
Relative → **portable**.


#### `script0d-crime-raw.py`
**Input reads found:**

1. `pd.read_csv("data/raw/crime_fbi.csv")`  
**Path type:**  
Relative → **portable**.


#### `script1-review-main-data.py`
**Input reads found:**

1. `pd.read_parquet("interim-dfs-copy/main_merged_raw.parquet")`
**Path type:**  
Relative → **portable**, but depends on outputs from scripts 0a–0d.


#### `script2-aggregate-merge.py`
**Input reads found:**

1. `pd.read_csv("/"todays_str_date"clean.csv")`
**Path type:**  
Relative → **portable**.


#### `script3-ridge.py`
**Input reads found:**

1. `pd.read_csv("interim-dfs-copy/analysis.csv")`


