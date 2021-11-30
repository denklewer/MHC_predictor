from os import listdir
from os.path import isfile, join
from mhcnames import normalize_allele_name
import pandas as pd


def get_available_alleles(data_path):
    result_alleles = []
    only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    per_allele_train_df = dict()
    for file_name in only_files:
        allele_name = "".join(file_name.split("_")[:4])
        allele_name = normalize_allele_name(allele_name)
        # per_allele_train_df[allele_name] =  pd.read_csv(DATA_PATH + file_name,sep=";")
        result_alleles.append(allele_name)
    result_alleles = list(set(result_alleles))
    return result_alleles


def read_data(data_path, alleles_to_read, dataset_type, data_type):
    assert data_type in ["binders", "nonbinders", 'immunogenic', 'nonimmunogenic']
    assert dataset_type in ["train", "test"]
    per_allele_per_kfold_per_length_peptides = dict()

    for allele_name in alleles_to_read:
        per_allele_per_kfold_per_length_peptides[allele_name] = dict()
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
        file_name = "{}_{}_{}.csv".format(target_allele_name, data_type, dataset_type)
        total_df = pd.read_csv(data_path + file_name,sep=";")
        n_splits = list(total_df['split'].unique())
        lengths = list(total_df['length'].unique())
        for split_num in n_splits:
            per_allele_per_kfold_per_length_peptides[allele_name][split_num] = dict()
            for length in lengths:
                per_allele_per_kfold_per_length_peptides[allele_name][split_num][length] = total_df[(total_df['length'] == length) &
                                                                                                    (total_df['split'] == split_num)]['peptide'].values
    return per_allele_per_kfold_per_length_peptides
