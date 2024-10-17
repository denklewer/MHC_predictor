from os import listdir
from os.path import isfile, join
import mhcgnomes
import mhcnames
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict


def get_available_alleles(data_path):
    result_alleles = []
    only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    per_allele_train_df = dict()
    for file_name in only_files:
        allele_name = file_name.split("_")[0]
        allele_name = mhcgnomes.parse(allele_name).to_string()
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
        target_allele_name = mhcnames.compact_allele_name(allele_name)
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


def merge_per_allele_sets_into_combination_sets(per_allele_per_kfold_per_length_binders_train,
                                                per_allele_per_kfold_per_length_binders_test,
                                                combination_size, selected_alelles, total_splits, list_of_transform_dicts=None):
    f"""
    works with result of {read_data}
    takes combination size and merges all datasets together in all possible wais using {itertools.combinations}
    :return:  per_allele_per_kfold_per_length_peptides, where allele is a combination of old alleles
    """
    per_allele_per_kfold_per_length_binders_train_new = dict()
    per_allele_per_kfold_per_length_binders_test_new = dict()
    OLD_ALLELES = selected_alelles
    NEW_ALELLES = list()
    TOTAL_SPLITS = total_splits

    for combination in itertools.combinations(OLD_ALLELES, combination_size):
        allele_name = "+".join(combination)
        NEW_ALELLES.append(allele_name)
        print(allele_name)
        per_allele_per_kfold_per_length_binders_train_new[allele_name] = dict()
        per_allele_per_kfold_per_length_binders_test_new[allele_name] = dict()
        for split_num in range(TOTAL_SPLITS):
            per_allele_per_kfold_per_length_binders_train_new[allele_name][split_num] = dict()
            per_allele_per_kfold_per_length_binders_test_new[allele_name][split_num] = dict()

            # for each split find intersected lengths
            list_of_length_sets = list()
            for old_allele in combination:
                lengths = set(per_allele_per_kfold_per_length_binders_train[old_allele][split_num].keys())
                list_of_length_sets.append(lengths)
            intersected_lengths = set.intersection(*list_of_length_sets)

            for target_length in intersected_lengths:
                data_arrays_train = list()
                data_arrays_test = list()
                if split_num == 0:
                    print(f"Length {target_length}:" , end=" ")
                for old_allele in combination:
                    single_allele_data_array_train = per_allele_per_kfold_per_length_binders_train[old_allele][split_num][target_length]
                    if list_of_transform_dicts:
                        single_allele_data_array_train = np.array([np.array([tuple(transform_dict[aa] for transform_dict in list_of_transform_dicts) for aa in peptide]) for peptide in single_allele_data_array_train])

                    if split_num == 0:
                        print(f"{old_allele} - {len(single_allele_data_array_train)}", end= " ")
                    data_arrays_train.append(single_allele_data_array_train)
                    single_allele_data_array_test = per_allele_per_kfold_per_length_binders_test[old_allele][split_num][target_length]
                    if list_of_transform_dicts:
                        single_allele_data_array_test = np.array([np.array([tuple(transform_dict[aa] for transform_dict in list_of_transform_dicts) for aa in peptide]) for peptide in single_allele_data_array_test])
                    if split_num == 0:
                        # print(f"{old_allele} - {len(single_allele_data_array_test)}", end= " ")
                        pass
                    data_arrays_test.append(single_allele_data_array_test)
                if split_num == 0:
                    print()

                per_allele_per_kfold_per_length_binders_train_new[allele_name][split_num][target_length] = np.concatenate(data_arrays_train)
                per_allele_per_kfold_per_length_binders_test_new[allele_name][split_num][target_length] = np.concatenate(data_arrays_test)
    return per_allele_per_kfold_per_length_binders_train_new, per_allele_per_kfold_per_length_binders_test_new

def extract_count_df(per_allele_per_kfold_per_length_binders, selected_alleles, total_splits ):
    f"""
    support method for calculating weights for input data.
    takes result of {read_data} as input. 
    :return: dataframe with counts for all lengths inside each allele
    """
    per_split_intersected_lengths = dict()
    for split_num in range(total_splits):
        list_of_length_sets = list()
        for old_allele in selected_alleles:
            lengths = set(per_allele_per_kfold_per_length_binders[old_allele][split_num].keys())
            list_of_length_sets.append(lengths)
        intersected_lengths = set.intersection(*list_of_length_sets)
        per_split_intersected_lengths[split_num] = intersected_lengths

    per_kfold_per_length_per_allele_counts = {
            split_num: {
                target_length: {
                    allele_name: len(per_allele_per_kfold_per_length_binders[allele_name][split_num][target_length])
                    for allele_name in selected_alleles
                } for target_length in per_split_intersected_lengths[split_num]
            } for split_num in range(total_splits)
        }

    result_dict = {}
    for kfold,  per_length_per_allele_counts in per_kfold_per_length_per_allele_counts.items():
        for length, per_allele_counts in per_length_per_allele_counts.items():
            for allele_name , count in  per_allele_counts.items():
                result_dict[(allele_name, kfold, length)] =  {'cnt': count }
    count_df = pd.DataFrame.from_dict(result_dict, orient="index")
    count_df.index.names = ['allele', 'split', 'length']
    return count_df

def calculate_weights_based_on_length_counts(count_df, selected_alleles, total_splits, selected_lengths, use_weights):
    f"""  
    :param count_df: count df for each single allele
    :param selected_alleles: combined alleles
    :param total_splits: 
    :param selected_lengths: 
    :param use_weights: 
    works with result of {extract_count_df}
    Just calculate portions of given length inside dataset if  {use_weights} is True, else assign 1 to each sequence
    :return: 
    """
    per_allele_per_kfold_per_length_weights = dict()
    ALLELES = selected_alleles

    for combined_allele_name in ALLELES:
        per_allele_per_kfold_per_length_weights[combined_allele_name] = dict()
        old_alleles = combined_allele_name.split('+')
        max_weight = 0
        for split_num in range(total_splits):
            per_allele_per_kfold_per_length_weights[combined_allele_name][split_num] = dict()
            sub_count_df_train = count_df.loc[(count_df.index.get_level_values('split') == 0) &
                                                    (count_df.index.get_level_values('allele').isin(old_alleles)) &
                                                    (count_df.index.get_level_values('length').isin(selected_lengths))]
            total_data_length = sub_count_df_train['cnt'].sum()
            print(f"Total data length for split {split_num} is {total_data_length}")
            for target_length in selected_lengths:
                current_weights_list = list()
                weights_dict = dict()
                for old_allele_name in old_alleles:
                    old_allele_cnt = count_df.loc[(old_allele_name, split_num, target_length)]['cnt']
                    current_weights_list.extend([total_data_length/old_allele_cnt if use_weights else 1 for i in range(old_allele_cnt)])
                    weights_dict[old_allele_name] = total_data_length/old_allele_cnt if use_weights else 1
                max_weight = max(max(current_weights_list),max_weight)
                per_allele_per_kfold_per_length_weights[combined_allele_name][split_num][target_length] = current_weights_list
                print(f"Weights for lentgh {target_length} are {weights_dict}")
            print("Devide by", max_weight)
            result_weight_dict = dict()
            for target_length in selected_lengths:
                current_weights_list =  per_allele_per_kfold_per_length_weights[combined_allele_name][split_num][target_length]
                current_weights_list = [item/max_weight for item in current_weights_list]
                per_allele_per_kfold_per_length_weights[combined_allele_name][split_num][target_length] = current_weights_list
                result_weight_dict[target_length] = np.unique(current_weights_list)
            print(f"Combination {combined_allele_name}, split {split_num}: {result_weight_dict}")

                
  
    return per_allele_per_kfold_per_length_weights







