from mhcnames import normalize_allele_name## ONLY FOR LINUX
import pandas as pd
import os
import sys
import numpy as np
from data_reading import read_genetech_data
from MHCSeqNet.PredictionModel.BindingSequencePredictor import BindingSequencePredictor
from MHCSeqNet.PredictionModel.Utility.SequenceBasedModel import SequenceBasedModel
from config import CLASSES_TO_USE, ORGANISMS_TO_USE, PER_ORGANISM_PER_CLASS_ALLELES, RESULT_COLUMNS

#rewrite classes to use to only I class because it's only class I method
METHOD = 'mhcseqnet'
ORGANISMS_TO_USE = ['Human']
CLASSES_TO_USE = ['I']
ALLELES = [item  for organism_to_use in ORGANISMS_TO_USE 
                 for class_to_use in CLASSES_TO_USE 
                 for item in PER_ORGANISM_PER_CLASS_ALLELES[organism_to_use][class_to_use]]


# importing the predict module

DATA_PATH = "/home/denklewer/Data/"
TARGET_PATH_TO_PREDICTIONS = DATA_PATH + 'predictions/'+ METHOD+'/'



per_allele_train_df, per_allele_val_df = read_genetech_data(DATA_PATH, CLASSES_TO_USE, ALLELES)

assert not 0 in [len(per_allele_val_df[allele_name]) for allele_name in ALLELES], 'One of dfs has length of 0, check data reading'

def defineClass(allele_name):
    for organism_to_use in ORGANISMS_TO_USE:
        for class_to_use in CLASSES_TO_USE:
            if allele_name in PER_ORGANISM_PER_CLASS_ALLELES[organism_to_use][class_to_use]:
                return class_to_use
            
# add ID, add length, filter by length, rename label column,add normalized allele
for allele_name in ALLELES:
    per_allele_val_df[allele_name]['source_sequence_name'] =  'seq' + per_allele_val_df[allele_name].index.astype(str).values
    target_allele = normalize_allele_name(allele_name)
    per_allele_val_df[allele_name]['mhc'] = target_allele
    per_allele_val_df[allele_name]['length'] = per_allele_val_df[allele_name]['peptide'].str.len()

#     #Filter for mhcnuggets <=15 length for class 1 and <=30 for class 2
    if defineClass(allele_name) == 'I':
        per_allele_val_df[allele_name] = per_allele_val_df[allele_name][per_allele_val_df[allele_name]['length'] <= 15]
#     else:
#         per_allele_val_df[allele_name] = per_allele_val_df[allele_name][per_allele_val_df[allele_name]['length'] <= 30]

PREDICTOR_HOME = os.getenv('MHCSEQNET_HOME')
print(f"Predictor home is {PREDICTOR_HOME}")
#make total df
total_df = pd.concat(per_allele_train_df.values())



# Initialize model instance
configuration = SequenceBasedModel.get_default_configuration()
configuration['path_to_allele_information'] = f'{PREDICTOR_HOME}/PretrainedModels/sequence_model/AlleleInformation.txt'

bindingSequencePredictor = BindingSequencePredictor()
# Utility/AlleleInformation.txt should exist
bindingSequencePredictor.train_model(
    'trained_models/MHCSeqNet/', 
    total_df.peptide.values,
    total_df.allele.values,
    total_df.binder.values,
    configuration=configuration,
    amino_acid_representation_path=f'{PREDICTOR_HOME}/MHCSeqNet/AminoAcidRepresentationModel/Model/embedding_weight_merged_5_d6.h5')