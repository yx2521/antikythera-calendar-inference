# S2_Coursework

A coursework project for the S2 module: using Bayesian inference to estimate the number of holes in the complete calendar ring of the Antikythera mechanism. Two different models are evaluated and compared.

## Author
- Yilan Xu

## Project Structure

- `src/`
  - `hole_model.py` – Custom functions used throughout the project  
  - `notebook_demonstration/` – Step-by-step notebook demonstrating the full workflow

- `plots/` – All figures used in the final report

- `reports/`
  - `S2_report.pdf` – Main report

- `dataverse_files/` – Source data used in the analysis


## Installation

To install required Python packages:

```bash
pip install -r requirements.txt
```

## Testing
Tests on JAX auto-differentiation are located in the tests/ folder and can be run with:
```bash
pytest tests/
```

## All the other necessary details are included in the report

## Declaration
During the development of this project, I made use of Github Copilot’s autocompletion feature to assist in coding and generating docstrings for functions. Additionally, ChatGPT was used to support debugging. For the final report, ChatGPT helped in generating LaTeX code in desired formats and organising/polishing written languages.