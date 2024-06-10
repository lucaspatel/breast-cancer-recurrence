import pandas as pd
import os


def load_dataset(data_path, counts_file, tpm_file):
    """
    Load counts data from specified paths based on file extensions.

    Parameters:
    data_path (str): Path to the directory containing the data file.
    counts_file (str): Filename of the counts data.
    tpm_file (str): Filename of the TPM data.

    Returns:
    tuple: DataFrame of counts, DataFrame of TPM data
    """

    def load_file(file_path):
        """
        Load a file based on its extension.
        Supports CSV and Excel files.
        """
        extension = os.path.splitext(file_path)[1].lower()
        if extension in [".xls", ".xlsx"]:
            return pd.read_excel(file_path, index_col=0)
        elif extension in [".csv"]:
            return pd.read_csv(file_path)
        else:
            return pd.read_csv(file_path, sep="\t")

    counts_path = os.path.join(data_path, counts_file)
    counts_data = load_file(counts_path)

    tpm_path = os.path.join(data_path, tpm_file)
    tpm_data = load_file(tpm_path)

    return counts_data, tpm_data


def load_metadata(data_path, metadata_file):
    """
    Load metadata from specified paths based on file extensions.

    Parameters:
    data_path (str): Path to the directory containing the metadata file.
    metadata_file (str): Filename of the metadata file.

    Returns:
    pd.DataFrame or None: Loaded metadata as a pandas DataFrame, or None if the file does not exist or an error occurs.
    """
    if not metadata_file:
        print("No metadata file specified.")
        return None

    metadata_path = os.path.join(data_path, metadata_file)

    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file {metadata_path} not found.")
        return None

    extension = os.path.splitext(metadata_path)[1].lower()
    if extension in [".xls", ".xlsx"]:
        return pd.read_excel(metadata_path, index_col=0)
    elif extension == ".csv":
        return pd.read_csv(metadata_path)
    else:
        print(f"Warning: Unsupported file format {extension} for file {metadata_path}.")
        return None


datasets = {
    "pnas_norm": (
        "pnas_normal_readcounts.txt",
        "pnas_normal_tpm.txt",
    ),
    "pnas_bc": (
        "pnas_readcounts_96_nodup.txt",
        "pnas_tpm_96_nodup.txt",
    ),
    "val": (
        "validation_exon_readcounts",
        "validation_exon_tpm",
    ),
}


def load_data():
    data = {}
    for dataset, files in datasets.items():
        data[dataset] = {}
        data[dataset]["counts"], data[dataset]["tpm"] = load_dataset("data/", *files)

    patient_info_metadata = load_metadata("data/data/", "pnas_patient_info.csv")
    val_bc_metadata = load_metadata("data/data/", "validation_bc_meta.xlsx")
    val_normal_metadata = load_metadata("data/data/", "validation_normal_meta.xlsx")

    # correct BC counts data
    original_headers = data["pnas_bc"][
        "counts"
    ].columns  # This holds what pandas thought were data
    new_row = pd.DataFrame([original_headers], columns=data["pnas_bc"]["counts"].columns)
    data["pnas_bc"]["counts"] = pd.concat(
        [new_row, data["pnas_bc"]["counts"]], ignore_index=True
    )
    new_headers = ["Index"] + [
        f"S{i:02}" for i in range(1, data["pnas_bc"]["counts"].shape[1])
    ]  # Adjust to account for all columns
    data["pnas_bc"]["counts"].columns = new_headers
    data["pnas_bc"]["counts"].set_index(data["pnas_bc"]["counts"].columns[0], inplace=True)
    data["pnas_bc"]["counts"].index.name = "Ensembl ID"

    # correct BC TPM data
    original_headers = data["pnas_bc"][
        "tpm"
    ].columns  # This holds what pandas thought were data
    new_row = pd.DataFrame([original_headers], columns=data["pnas_bc"]["tpm"].columns)
    data["pnas_bc"]["tpm"] = pd.concat([new_row, data["pnas_bc"]["tpm"]], ignore_index=True)
    new_headers = ["Index"] + [
        f"S{i:02}" for i in range(1, data["pnas_bc"]["tpm"].shape[1])
    ]  # Adjust to account for all columns
    data["pnas_bc"]["tpm"].columns = new_headers
    data["pnas_bc"]["tpm"].set_index(data["pnas_bc"]["tpm"].columns[0], inplace=True)
    data["pnas_bc"]["tpm"].index.name = "Ensembl ID"

    # correct normal counts data
    data["pnas_norm"]["counts"].index.name = "Ensembl ID"

    # correct normal TPM data
    data["pnas_norm"]["tpm"].index.name = "Ensembl ID"

    # correct validation counts data
    data["val"]["counts"].index.name = "Ensembl ID"

    # correct validation TPM data
    data["val"]["tpm"].index.name = "Ensembl ID"

    # merge tables
    counts_table = data["pnas_bc"]["counts"].join(data["pnas_norm"]["counts"], how="left")
    tpm_table = data["pnas_bc"]["tpm"].join(data["pnas_norm"]["tpm"], how="left")

    # add metadata
    patient_info_metadata.index = patient_info_metadata["sample_id"].apply(
        lambda x: x.split("_")[0]
    )
    counts_table_transposed = counts_table.T
    counts_table = counts_table_transposed.join(patient_info_metadata, how="left")
    tpm_table_transposed = tpm_table.T
    tpm_table = tpm_table_transposed.join(patient_info_metadata, how="left")

    # repeat for validation data
    val_counts_table = data["val"]["counts"]
    val_tpm_table = data["val"]["tpm"]

    val_normal_metadata.rename(
        columns={"Gender (M/F)": "Gender", "Age (in years)": "Age"}, inplace=True
    )
    val_bc_metadata.rename(columns={"Age at Sample collection": "Age"}, inplace=True)
    val_bc_metadata["Gender"] = val_bc_metadata["Gender"].replace(
        {"Male": "M", "Female": "F"}
    )
    val_combined_metadata = pd.concat(
        [val_normal_metadata, val_bc_metadata], ignore_index=False
    )

    val_counts_table_transposed = val_counts_table.T
    val_counts_table = val_counts_table_transposed.join(val_combined_metadata, how="left")
    val_tpm_table_transposed = val_tpm_table.T
    val_tpm_table = val_tpm_table_transposed.join(val_combined_metadata, how="left")
    
    return tpm_table, counts_table, val_tpm_table, val_counts_table
