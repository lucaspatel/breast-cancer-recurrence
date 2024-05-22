# Alzheimer's Disease Prediction

This is an analysis of Alzheimer's Disease (AD) gene expression data from three independent studies.

This is companion code to the final project for BENG 203 / CSE 283 (SP24) authored by **Daira Melendez** (), **Raimon Padrós I Valls** (A59025488), and **Lucas Patel** (A13041630).

## Getting Started

### Prerequisites

This analysis is performed in Python and requires several dependencies. A premade conda environment is provided for convenience, or packages can be installed manually. If installing conda for the first time, we suggest using [Miniconda](https://docs.anaconda.com/free/miniconda/) as a lightweight installer. 

### Installation

Follow these steps to install and configure the repository:

1. Clone the repository:
    ```bash
    git clone https://github.com/lucaspatel/cse283-final
    ```
2. Change directory to the cloned respository:
    ```bash
    cd cse283-final
    ```
3. Next, configure the conda environment and other dependencies as follows:
    ```bash
    make setup
    ```
4. Activate the conda environment:
   ```bash
   conda activate ad-prediction
   ```

## Usage
Usage may vary depending on use case.

```python
```

## Development

## Troubleshooting
> I'm having trouble with the `make setup` command. Specifically, my terminal just says "Solving Environment" and never finishes!
This is a well-known problem with conda. If you have [mamba](https://mamba.readthedocs.io/en/latest/) installed, you can try directly configuring the environment:
```bash
mamba env create --name admix -f environment.yml
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
Data is derived from [Yan et al.](https://www.cell.com/current-biology/fulltext/S0960-9822(20)30291-8), [Toden et al.](https://www.science.org/doi/10.1126/sciadv.abb1654), and [Burgos et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094839) and curated by [the Zhong Lab](https://github.com/Zhong-Lab-UCSD/AD_prediction_blood).