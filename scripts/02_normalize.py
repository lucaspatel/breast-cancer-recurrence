"""
Normalize gene expression data using rnanorm.

Date: 2024-05-22
"""

import pandas as pd


def load_data(data_path, counts_file, metadata_file):
    """
    Load counts data and metadata from specified paths.

    Parameters:
    data_path (str): Path to the data directory.
    counts_file (str): Filename of the counts data.
    metadata_file (str): Filename of the metadata.

    Returns:
    tuple: DataFrame of counts, DataFrame of metadata
    """

    counts_path = f"{data_path}/{counts_file}"
    counts_data = pd.read_csv(counts_path, sep="\t")

    metadata_path = f"{data_path}/{metadata_file}"
    metadata = pd.read_excel(metadata_path, index_col=0)

    return counts_data, metadata


datasets = {
    "burgos_dbgap": ("burgos_dbgap_counts.txt", "burgos_dbgap_metadata.xlsx"),
    "silver_seq": ("silver_seq_counts.txt", "silver_seq_metadata.xlsx"),
    "toden": ("toden_counts.txt", "toden_metadata.xlsx"),
}

data = {}
for dataset, files in datasets.items():
    data[dataset] = {}
    data[dataset]["counts"], data[dataset]["metadata"] = load_data(
        f"data/{dataset}", *files
    )
