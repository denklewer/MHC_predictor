CLASSES_TO_USE = ['I', 'II']
ORGANISMS_TO_USE = ['Human','Mouse']


PER_ORGANISM_PER_CLASS_ALLELES = {
    'Human': {
        'I': ['HLA_A_02_01','HLA_B_27_05', "HLA_A_03_01", "HLA_B_07_02"],
        'II': [
#                'HLA_DR', 
               'HLA_DRB1_01_01', 'HLA_DRB1_04_01', 'HLA_DRB1_04_05'
#             'HLA_DQ'
        ]
    },
    'Mouse': {
        'I':['H2_Db','H2_Kb'],
        'II': ['H2_IAb', 'H2_IAd', 
               #'H2_d class II', 'H2_b class II'
              ]
    }
    
}

RESULT_COLUMNS = {
    "label":"binder",
    "peptide":"peptide",
    "id":"source_sequence_name"
}
