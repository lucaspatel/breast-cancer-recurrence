import pandas as pd
import os


def load_data(data_path, counts_file, tpm_file, metadata_file):
    """
    Load counts data and metadata from specified paths.

    Parameters:
    data_path (str): Path to the data directory.
    counts_file (str): Filename of the counts data.
    tpm_file (str): Filename of the TPM data.
    metadata_file (str): Filename of the metadata. Can be an empty string if not applicable.

    Returns:
    tuple: DataFrame of counts, DataFrame of TPM data, DataFrame of metadata (or None if metadata file is not provided)
    """

    counts_path = os.path.join(data_path, counts_file)
    counts_data = pd.read_csv(counts_path, sep="\t")

    tpm_path = os.path.join(data_path, tpm_file)
    tpm_data = pd.read_csv(tpm_path, sep="\t")

    if metadata_file:
        metadata_path = os.path.join(data_path, metadata_file)
        if os.path.exists(metadata_path):
            metadata = pd.read_excel(metadata_path, index_col=0)
        else:
            print(f"Warning: Metadata file {metadata_path} not found.")
            metadata = None
    else:
        print("No metadata file specified.")
        metadata = None

    return counts_data, tpm_data, metadata


datasets = {
    "pnas_bc": (
        "pnas_normal_readcounts.txt",
        "pnas_normal_tpm.txt",
        "data/validation_bc_meta.xlsx",
    ),
    "pnas_norm": (
        "pnas_readcounts_96_nodup.txt",
        "pnas_tpm_96_nodup.txt",
        "data/validation_normal_meta.xlsx",
    ),
    "val": (
        "validation_exon_readcounts",
        "validation_exon_tpm",
        "",
    ),
}

data = {}
for dataset, files in datasets.items():
    data[dataset] = {}
    data[dataset]["counts"], data[dataset]["tpm"], data[dataset]["metadata"] = (
        load_data("data/", *files)
    )
