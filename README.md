# Breast Cancer Recurrence Prediction

This is an analysis of breast cancer SILVER-seq gene expression data from [Zhou et al.](https://www.pnas.org/doi/10.1073/pnas.1908252116).

This is companion code to the final project for BENG 203 / CSE 283 (SP24) authored by **Daira Melendez** (), **Raimon Padrós I Valls** (A59025488), and **Lucas Patel** (A13041630).

## Getting Started

### Prerequisites

This analysis is performed in Python and requires several dependencies. A premade conda environment is provided for convenience, or packages can be installed manually. If installing conda for the first time, we suggest using [Miniconda](https://docs.anaconda.com/free/miniconda/) as a lightweight installer. 

### Installation

Follow these steps to install and configure the repository:

1. Clone the repository:
    ```bash
    git clone https://github.com/lucaspatel/breast-cancer-recurrence
    ```
2. Change directory to the cloned respository:
    ```bash
    cd breast-cancer-recurrence
    ```
3. Next, configure the conda environment and other dependencies with `make`. This step will also download all the required datasets:
    ```bash
    make setup
    ```
4. Activate the conda environment:
   ```bash
   conda activate beng203
   ```

## Usage
Usage is not yet determined. Consider exploring the `scripts` directory or the `explore.ipynb` notebook.

```python
```

## Development
Currently...

## Troubleshooting
> I'm having trouble with the `make setup` command. Specifically, my terminal just says "Solving Environment" and never finishes!

This is a well-known problem with conda. If you have [mamba](https://mamba.readthedocs.io/en/latest/) installed, you can try directly configuring the environment:
```bash
mamba env create --name beng203 -f environment.yml
```

## Contributing
We welcome contributions from the community. If you're interested in improving this work, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch and open a pull request.

## License

This project is licensed under the MIT License. For more details, see the `LICENSE` file in the project repository.

## Acknowledgments
Data is derived from [Zhou et al.](https://www.pnas.org/doi/10.1073/pnas.1908252116) and curated by [the Zhong Lab](https://github.com/Zhong-Lab-UCSD/breast_cancer_recurrence_classifier).
