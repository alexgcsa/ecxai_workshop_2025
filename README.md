# Interpreting Machine Learning Pipelines Produced by Evolutionary AutoML for Biochemical Property Prediction
### By [Alex G. C. de Sá](https://scholar.google.com/citations?user=K572cZ0AAAAJ), [Gisele L. Pappa](https://scholar.google.com/citations?user=C_0ZLuYAAAAJ), [Alex A. Freitas](https://scholar.google.com/citations?user=NEP3RPYAAAAJ&hl=en) and [David B. Ascher](https://scholar.google.co.uk/citations?user=7KrAVc0AAAAJ&hl=en)
### Code for the paper accepted for the workshop [Evolutionary Computing and Explainable Artificial Intelligence](https://ecxai.github.io/ecxai/workshop-2025) at the [GECCO conference](https://gecco-2025.sigevo.org/HomePage).


## 📦 Project Structure

```
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks for experiments
├── scripts/              # Scripts for data processing and modeling
├── requirements.yml      # Conda environment specification
└── README.md             # Project documentation
```

## 🛠️ Environment Setup

To set up the project environment using Conda, follow the steps below:

### 1. Create the conda environment

Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

Then run:

```bash
conda env create -f requirements.yml
```

### 2. Activate the environment

```bash
conda activate automl_biochem
```

### 3. Deactivate the environment (when you're done)

```bash
conda deactivate
```

## 📖 Usage

Once the environment is set up, you can run the provided notebooks and scripts to reproduce results or start new experiments.

## 🧪 Requirements

All required dependencies are listed in `requirements.yml`. The environment includes packages for data processing, machine learning, and biochemical analysis.

## 📬 Contact

For questions or contributions, please open an [issue](https://github.com/yourusername/yourrepo/issues) or contact the maintainer.
