# Fake News Detection

## Repository Overview

This repository supports the Fake News Detection project by Team Thunder Pandas. The goal is to classify political statements into six levels of truthfulness using machine learning on the LIAR dataset. This repository is organized for reproducibility and clarity, following best practices for open data science projects. It contains all scripts, data documentation, outputs, and supporting files needed to understand, reproduce, and extend the analyses.

---

## Software and Platform

This project was developed and tested using Python 3 in Google Colab (cloud Linux environment), but can be run locally on any modern Windows, Mac, or Linux machine with the necessary Python packages installed.

**Primary software and libraries:**
- Python 3.8+ (tested on Google Colab)
- numpy
- pandas
- scikit-learn
- matplotlib
- torch
- transformers
- tqdm

To set up all dependencies, refer to `requirements.txt` if provided, or manually install these packages using pip:

------------------------------------------------------------------------------------

## Documentation

### Repository Structure

Below is an outline of the folder structure and contents:


├── README.md # Project orientation and reproduction instructions


├── LICENSE.md # MIT License for all code in this repository


├── SCRIPTS/ # All source code scripts

│ ├──DS4002_Complete_Project1_FakeNewsClassification_ThunderPandas.ipynb

│ ├──bert_binary_model.py

│ ├──adding data into scripts for and adding scripts

│ ├──bert_multiclass_model.py

│ ├──requirements.txt

│ ├──tf-idf_baseline_model.py


├── DATA/


│ ├── READMEdata.md

│ ├── test-test-clean.csv

│ ├── test.tsv

│ ├── train-train-clean.csv

│ ├── train.tsv

│ ├── valid-valid-clean.csv

│ ├── valid.tsv



├── OUTPUT/

│ ├──  confusion_matrix_baseline_bert_model.png

│ ├── confusion_matrix_binary_baseline_bert.png

│ ├── confusion_matrix_binary_combined_bert_metadata.png

│ ├── confusion_matrix_combined_bert_metadata.png

│ ├── confusion_matrix_tf-idf.png


└── REFERENCES.md # Full list of literature and code citations



## Reproduction

To reproduce the results in this repository, follow these steps:

1. **Clone this repository:**

git clone https://github.com/SanjayKarun4444/Fake-News-Detection-CS4002-ThunderPandas.git

cd Fake-News-Detection-CS4002-ThunderPandas


2. **Set up the Python environment:**
- If `requirements.txt` or `environment.yml` are provided, run:
  ```
  pip install -r requirements.txt
  ```
  *or*
  ```
  conda env create -f environment.yml
  conda activate [ENV_NAME]
  ```

- Otherwise, install dependencies manually (see Software and Platform).

3. **Acquire the data:**
- If the `liar_dataset.csv` is present in the `DATA/` folder, no action is needed.
- If the data is too large to store on GitHub, follow the instructions in `DATA/GET_DATA.md` to download the original LIAR dataset from [Activeloop Hub](https://datasets.activeloop.ai/docs/ml/datasets/liar-dataset/).

4. **Run all scripts in order:**
- Change into the `SCRIPTS/` directory, and run each numbered script sequentially to clean the data, engineer features, train models, and evaluate.
  ```
  python 01_data_cleaning.py
  python 02_feature_engineering.py
  python 03_model_logistic_regression.py
  python 04_model_distilbert.py
  python 05_evaluation.py
  ```
- Each script is heavily commented and includes header docstrings describing functionality and expected inputs/outputs.

5. **View Results:**
- All output files (accuracy scores, plots, and tables) will be automatically generated in the `OUTPUT/` folder.
- Refer to `OUTPUT/` for confusion matrices, key visualizations, and numerical evaluation.

6. **Data Documentation and Metadata:**
- See `DATA/liar_metadata.md` for full descriptions of the dataset, its provenance, ethical considerations, data dictionary, and visualizations.
- All plots used in exploratory analysis are included for reviewers.

7. **References:**
- Consult `REFERENCES.md` at the repository root for a full list of references, including datasets, modeling papers, and code sources.

---

For any difficulties, please open an Issue or refer to the scripts’ inline instructions.
