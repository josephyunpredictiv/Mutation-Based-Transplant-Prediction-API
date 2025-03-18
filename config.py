import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Neural Network Path
#model_path_old = BASE_DIR / "artifacts/prognosis_model_biochemical_amino_acids.pth"
model_path = BASE_DIR / "artifacts/prognosis_model_biochemical_amino_acids_unknown_token.pth"

# R Preprocessing Script Path
R_path = BASE_DIR / "data_preprocessing/vcf_preprocessing.R"
# File Upload
ALLOWED_EXTENSIONS = {"txt", "csv", "vcf"}
REQUIRED_COLUMNS = [
    "Gene", "Clinical_significance_score_per_variant", "Protein_function_impairment_score_per_variant",
    "Disease_name", "Chromosome", "Position", "Zygosity", "Variant_type",
    "Base_change", "Amino_acid_change",
    "ExAC_ALL", "ExAC_AFR", "ExAC_AMR", "ExAC_EAS", "ExAC_FIN",
    "ExAC_NFE", "ExAC_OTH", "ExAC_SAS", "AF"
]

# Data Directories
DATA_DIR = BASE_DIR / "data"
raw = DATA_DIR / "raw"
processed = DATA_DIR / "processed"
output = DATA_DIR / "output"

# Nonimportant Mutations
nonimportant_mutations = processed / "nonimportant_mutations.csv"

# Encoders
ENCODER_DIR = BASE_DIR / "artifacts/encoders"
aa_to_index = ENCODER_DIR / "aa_to_index.pkl"
base_to_index = ENCODER_DIR / "base_to_index.pkl"
biochemical_properties = ENCODER_DIR / "biochemical_properties.pkl"
chromosome_to_index = ENCODER_DIR / "chromosome_to_index.pkl"
disease_to_index = ENCODER_DIR / "disease_to_index.pkl"
gene_to_index = ENCODER_DIR / "gene_to_index.pkl"
unknown_disease_index = ENCODER_DIR / "unknown_disease_index.pkl"
unknown_gene_index = ENCODER_DIR / "unknown_gene_index.pkl"
variant_to_index = ENCODER_DIR / "variant_to_index.pkl"

# Scalers
SCALER_DIR = BASE_DIR / "artifacts/scalers"
scaler_allele = SCALER_DIR / "scaler_allele.pkl"
scaler_amino_acid = SCALER_DIR / "scaler_amino_acid.pkl"
scaler_position = SCALER_DIR / "scaler_position.pkl"
