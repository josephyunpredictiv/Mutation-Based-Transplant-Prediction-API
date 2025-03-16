import subprocess
import config
import os

def run_preprocessing(sample1_path,sample2_path,unimportant_path,output_path,pair_id):
    file_path=os.path.join(output_path,f"donor_patient_pair_{pair_id}.csv")
    subprocess.run(["Rscript", config.R_path, sample1_path, sample2_path, unimportant_path, file_path], check=True)
    return file_path