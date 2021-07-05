import pandas as pd
from mhcnames import normalize_allele_name## ONLY FOR LINUX
from config import RESULT_COLUMNS



def make_label_column(df, data_class):
    df.loc[df['el'] == 1, RESULT_COLUMNS['label']] = 1
    df.loc[df['el'] == 0, RESULT_COLUMNS['label']] = 0
    if data_class == 'II':
        df.loc[(df['data_type'] == 'BA') & (df['ic50'] <= 500), 'binder'] = 1
        df.loc[(df['data_type'] == 'BA') & (df['ic50'] > 500),'binder'] = 0
    df['binder'] = df['binder'].astype(int)
    return df


def read_genetech_data(DATA_PATH, CLASSES_TO_USE, ALLELES):
    GENENTECH_DATA_PATH = DATA_PATH + 'genentech/'
    per_class_per_split_data = dict()
    per_allele_train_df = dict()
    per_allele_test_df = dict()
    for class_to_use in CLASSES_TO_USE:
        per_class_per_split_data[class_to_use] = dict()
        
        class_data = pd.read_csv("{}gne_benchmark_class-{}.csv".format(GENENTECH_DATA_PATH, class_to_use))
        # unify. EL-> el
        class_data = class_data.rename(columns={'EL':'el'})
       
         
        train_data = class_data[class_data['split'] == 'train'].copy()
        test_data = class_data[class_data['split'] == 'test'].copy()
        
        # mark binders
        train_data = make_label_column(train_data, class_to_use)
        test_data = make_label_column(test_data, class_to_use)
        
        if class_to_use == 'I':
            allele_column = 'allele'
        else:
            allele_column = 'allotype'
        #normalize allele_column
        train_data[allele_column] = train_data[allele_column].str.replace('___', '-').apply(normalize_allele_name)
        test_data[allele_column] = test_data[allele_column].str.replace('___', '-').apply(normalize_allele_name)

        unique_alleles = train_data[allele_column].unique()
        for allele_name in ALLELES:
            if normalize_allele_name(allele_name) in unique_alleles:
                    per_allele_train_df[allele_name] = train_data[train_data[allele_column] == normalize_allele_name(allele_name)].copy() 
                    per_allele_test_df[allele_name] = test_data[test_data[allele_column] == normalize_allele_name(allele_name)].copy() 
    return per_allele_train_df, per_allele_test_df
        
    