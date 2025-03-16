import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from models import PrognosisNN

import re
import statistics
import joblib

import config

# Load pretrained Scalars
scaler_position = joblib.load(config.scaler_position)
scaler_allele = joblib.load(config.scaler_allele)
scaler_amino_acid = joblib.load(config.scaler_amino_acid)

# Load encoders
gene_to_index = joblib.load(config.gene_to_index)
disease_to_index = joblib.load(config.disease_to_index)
chromosome_to_index = joblib.load(config.chromosome_to_index)
variant_to_index = joblib.load(config.variant_to_index)
base_to_index = joblib.load(config.base_to_index)
aa_to_index = joblib.load(config.aa_to_index)

# Load UNKNOWN_INDEX values
UNKNOWN_GENE_INDEX = joblib.load(config.unknown_gene_index)
UNKNOWN_DISEASE_INDEX = joblib.load(config.unknown_disease_index)

# Load Biochemical Properties
biochemical_properties = joblib.load(config.biochemical_properties)


def evaluate_model(model_file_path, testing_data_file_path, pair_id='pair'):
  # Load new data
  testing_data = pd.read_csv(testing_data_file_path)

  # 1. Gene Index
  testing_data["Gene_Index"] = testing_data["Gene"].map(gene_to_index).fillna(UNKNOWN_GENE_INDEX).astype(int)

  # 2. Clinical_significance_score_per_variant & Protein_function_impairment_score_per_variant
  #testing_data["Clinical_significance_score_per_variant"].fillna(-1, inplace=True)
  #testing_data["Protein_function_impairment_score_per_variant"].fillna(-1, inplace=True)
  testing_data.fillna({"Clinical_significance_score_per_variant": -1,
                     "Protein_function_impairment_score_per_variant": -1}, inplace=True)


  # 3. Disease_name
  testing_data["Disease_Index"] = testing_data["Disease_name"].map(disease_to_index).fillna(UNKNOWN_DISEASE_INDEX).astype(int)

  # 4. Chromosome/Position
  testing_data["Chromosome_Index"] = testing_data["Chromosome"].map(chromosome_to_index)
  testing_data["Normalized_Position"] = testing_data["Position"] / 250_000_000

  # 5. Zygosity
  testing_data["Zygosity"] = testing_data["Zygosity"].map({"Heterozygous": 0, "Homozygous": 1})

  # 6. Variant_type
  testing_data["Variant_Index"] = testing_data["Variant_type"].map(variant_to_index)

  # 7. Base_change
  testing_data[["Ref_Base", "Mutation_Position", "Mut_Base"]] = testing_data["Base_change"].apply(lambda x: pd.Series(parse_coding_change(x)))
  testing_data["Mutation_Position"] = testing_data["Mutation_Position"].replace(-1, np.nan).fillna(-1).astype(int)
  testing_data["Ref_Base_Index"] = testing_data["Ref_Base"].map(base_to_index).astype(int)
  testing_data["Mut_Base_Index"] = testing_data["Mut_Base"].map(base_to_index).astype(int)

  # 8. Amino_acid_change
  testing_data[["Orig_AA", "Mutation_Position", "Mut_AA"]] = testing_data["Amino_acid_change"].apply(lambda x: pd.Series(parse_amino_acid_change(x)))
  
  # Biochemical properties dictionary for amino acids
  # Apply biochemical encoding
  testing_data["Orig_AA_Props"] = testing_data["Orig_AA"].map(biochemical_properties)
  testing_data["Mut_AA_Props"] = testing_data["Mut_AA"].map(biochemical_properties)
  testing_data["Orig_AA_Props"] = testing_data["Orig_AA_Props"].apply(lambda x: x if isinstance(x, list) else [0.0] * 5)
  testing_data["Mut_AA_Props"] = testing_data["Mut_AA_Props"].apply(lambda x: x if isinstance(x, list) else [0.0] * 5)

  # 9. Allele Frequency
  testing_data[["ExAC_ALL", "ExAC_AFR", "ExAC_AMR", "ExAC_EAS", "ExAC_FIN", "ExAC_NFE", "ExAC_OTH", "ExAC_SAS", "AF"]] = \
      testing_data[["ExAC_ALL", "ExAC_AFR", "ExAC_AMR", "ExAC_EAS", "ExAC_FIN", "ExAC_NFE", "ExAC_OTH", "ExAC_SAS", "AF"]].fillna(-1)

  testing_data["allele_frequency"] = testing_data[["ExAC_ALL", "ExAC_AFR", "ExAC_AMR", "ExAC_EAS", "ExAC_FIN", "ExAC_NFE", "ExAC_OTH", "ExAC_SAS", "AF"]].apply(lambda row: np.array(row, dtype=np.float32), axis=1)

  # 10. Normalize numerical features
  testing_data["Normalized_Position"] = scaler_position.transform(testing_data[["Normalized_Position"]])
  
  allele_freq_array = np.stack(testing_data["allele_frequency"].tolist())
  testing_data["allele_frequency"] = list(scaler_allele.transform(allele_freq_array))

  testing_data["Orig_AA_Props"] = list(scaler_amino_acid.transform(testing_data["Orig_AA_Props"].tolist()))
  testing_data["Mut_AA_Props"] = list(scaler_amino_acid.transform(testing_data["Mut_AA_Props"].tolist()))
  

  # Convert to PyTorch tensors
  gene_tensor = torch.tensor(testing_data["Gene_Index"].values, dtype=torch.long)
  disease_tensor = torch.tensor(testing_data["Disease_Index"].values, dtype=torch.long)
  chromosome_tensor = torch.tensor(testing_data["Chromosome_Index"].values, dtype=torch.long)
  variant_tensor = torch.tensor(testing_data["Variant_Index"].values, dtype=torch.long)
  ref_base_tensor = torch.tensor(testing_data["Ref_Base_Index"].values, dtype=torch.long)
  mut_base_tensor = torch.tensor(testing_data["Mut_Base_Index"].values, dtype=torch.long)

  position_tensor = torch.tensor(testing_data["Normalized_Position"].values, dtype=torch.float32)
  zygosity_tensor = torch.tensor(testing_data["Zygosity"].values, dtype=torch.float32)
  allele_freq_tensor = torch.tensor(np.stack(testing_data["allele_frequency"].tolist()), dtype=torch.float32)

  # Convert biochemical properties to tensors
  aa_orig_props_tensor = torch.tensor(np.stack(testing_data["Orig_AA_Props"].tolist()), dtype=torch.float32)  # Ensure proper shape
  aa_mut_props_tensor = torch.tensor(np.stack(testing_data["Mut_AA_Props"].tolist()), dtype=torch.float32)

  # Concatenate all features into a single tensor
  features_test = torch.cat([
      gene_tensor.unsqueeze(1), disease_tensor.unsqueeze(1), chromosome_tensor.unsqueeze(1),
      variant_tensor.unsqueeze(1), ref_base_tensor.unsqueeze(1), mut_base_tensor.unsqueeze(1),
      aa_orig_props_tensor, aa_mut_props_tensor,
      position_tensor.unsqueeze(1), zygosity_tensor.unsqueeze(1), allele_freq_tensor
  ], dim=1)

  model = PrognosisNN(6615, 812, 24, 9, 5, 5)
  model.load_state_dict(torch.load(model_file_path, weights_only=True))
  model.eval() 

  # Run predictions
  with torch.no_grad():
      (
          gene_t, disease_t, chrom_t, variant_t, ref_t, mut_t,
          aa_orig_t, aa_mut_t, allele_t, pos_t, zyg_t
      ) = torch.split(features_test, [1, 1, 1, 1, 1, 1, 1, 1, 9, 5, 5], dim=1)

      gene_t = gene_t.squeeze(1).long()
      disease_t = disease_t.squeeze(1).long()
      chrom_t = chrom_t.squeeze(1).long()
      variant_t = variant_t.squeeze(1).long()
      ref_t = ref_t.squeeze(1).long()
      mut_t = mut_t.squeeze(1).long()
      aa_orig_t = aa_orig_t.squeeze(1).long()
      aa_mut_t = aa_mut_t.squeeze(1).long()

      pos_t = pos_t.squeeze(1).float()
      zyg_t = zyg_t.squeeze(1).float()
      allele_t = allele_t.float()

      predictions = model(gene_t, disease_t, chrom_t, variant_t, ref_t, mut_t, aa_orig_t, aa_mut_t, pos_t, zyg_t, allele_t)

  # Convert probabilities to binary labels
  #pred_labels = (predictions.cpu().numpy().flatten() >= 0.5).astype(int)
  pred_labels = predictions.cpu().numpy().flatten()

  # Convert predictions to DataFrame
  testing_data["Predicted_Prognosis"] = pred_labels
  output_path=os.path.join(config.output,f"predicted_results_{pair_id}.csv")
  testing_data.to_csv(output_path, index=False)

  avg = statistics.mean(pred_labels)
  avg = np.float32(avg)
  #print("0=good 1=bad avg: " + str(avg))
  print(pair_id,str(avg))
  return avg

# Define useful parsing functions
def parse_coding_change(c_change):
    if not isinstance(c_change, str):  # Handle NaNs or non-string values
        return "UNKNOWN", -1, "UNKNOWN"
    match = re.match(r"c\.([ACGT])(\d+)([ACGT])", c_change)
    if match:
        ref, pos, mut = match.groups()
        return ref, int(pos), mut
    else:
        return "UNKNOWN", -1, "UNKNOWN"

def parse_amino_acid_change(aa_change):
    if not isinstance(aa_change, str):
        return "UNKNOWN", -1, "UNKNOWN"
    match = re.match(r"p\.([A-Z])(\d+)([A-Z])", aa_change)
    if match:
        ref, pos, mut = match.groups()
        return ref, int(pos), mut
    else:
        return "UNKNOWN", -1, "UNKNOWN"