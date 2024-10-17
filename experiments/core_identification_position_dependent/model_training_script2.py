import numpy as np
import pandas as pd
import sys
from data_reading_methods import get_available_alleles, read_data, merge_per_allele_sets_into_combination_sets
from data_reading_methods import extract_count_df, calculate_weights_based_on_length_counts
from hmm_visualization_methods import *
from hmm_logic_methods import train_model, save_model, init_grouped_structure_model_joiners_and_cycles_here




PATH_TO_PREDICTOR_HOME = "../.."
sys.path.append(PATH_TO_PREDICTOR_HOME)
METHOD = "hmm_pomegranate"

###############################
### Read data
###############################
DATA_PATH_BASE = 'G:/My Drive/Work/ITMO/data'
# DATA_PATH_BASE = 'D:/Google Drive/Work/ITMO/data' #windows
# DATA_PATH_BASE = '/home/denklewer/Data' #ubuntu

DATA_PATH = f'{DATA_PATH_BASE}/core_identification_paper/per_length_per_kfold_split/'  # work laptop
TARGET_PATH_TO_RESULTS = "experiment_results/core_identification_paper/"  # !CHANGE for bwv!

ALLELES = get_available_alleles(DATA_PATH)

per_allele_per_kfold_per_length_binders_train = read_data(DATA_PATH, ALLELES, "train", "binders")
per_allele_per_kfold_per_length_binders_test = read_data(DATA_PATH, ALLELES, "test", "binders")

###############################
### Define training params
###############################
# Training params
algorithm = 'viterbi'
combination_size = 2
TOTAL_SPLITS = 5
NUM_RUNS = 50
USE_WEIGHTS = False
check_peptides = True
target_class = "class1"

params = dict()

DataParams = {
    "class1": {
        "ALLELES": sorted(['HLA-A*01:01', 'HLA-A*03:01']),
        "TARGET_LENGTHS": [8, 9, 10, 11, 12],
        'start_cycle': False,
        'end_cycle': False,
        'intermediate_cycles': True,
        'emission_pseudocount': 50,
        'transition_pseudocount': 0,
        'cycle_pseudocount': 30,
        'anchor_top_aas': 3,
        'anchor_default_pseudocount': 50
    },

    "class2": {
        "ALLELES": sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 100,
        'transition_pseudocount': 50,
        'cycle_pseudocount': 300,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 50
    }

}

params['st'] = 25
params['alg'] = algorithm
params['min_iters'] = 30
params['max_iters'] = 90
params['emission_pseudocount'] = DataParams[target_class]["emission_pseudocount"]
params['transition_pseudocount'] = DataParams[target_class]["transition_pseudocount"]
params['verbose'] = True
params['lr_decay'] = 0.0
params['edge_inertia'] = 0.5
params['distribution_inertia'] = 0.5
params['groups'] = 5
params['states_per_group'] = 5
params['model_complexity'] = 2
params['use_pseudocount'] = True
# Class 1 params
params['self_cycle'] = False
params['start_cycle'] = DataParams[target_class]["start_cycle"]
params['end_cycle'] = DataParams[target_class]["end_cycle"]
params['equal_probs_for_cycle'] = False

params['freeze_distr'] = False  # wether to freeze start and end cycle distributions
params['tie_cycle_states'] = False
params['cycle_pseudocount'] = DataParams[target_class]["cycle_pseudocount"]
params['anchor_top_aas'] = DataParams[target_class]["anchor_top_aas"]
params['anchor_states'] = True
params['anchor_default_pseudocount'] = DataParams[target_class]["anchor_default_pseudocount"]
params['tie_anchor_states'] = False
params['intermediate_cycles'] = DataParams[target_class]["intermediate_cycles"]
params['equal_probs_for_intermediate_cycle'] = False
params['tie_intermediate_cycles'] = False
params['freeze_int_cycle_distr'] = False
params['position_component'] = False
params['multiple_check_input'] = False
ALLELES = DataParams[target_class]["ALLELES"]
TARGET_LENGTHS = DataParams[target_class]["TARGET_LENGTHS"]

assert params['st'] % params['groups'] == 0
assert params['st'] % params['states_per_group'] == 0
assert params['st'] == params['groups'] * params['states_per_group']

params_for_string = {key: value for key, value in params.items() if key in ['st', 'alg', 'min_iters']}
params_string = "-".join([f"{value}-{key}" for key, value in params_for_string.items()])
lengths_string = "-".join(map(str, TARGET_LENGTHS))

initial_params = None

###############################
### Transform data to desired form (for distributions)
###############################


per_allele_per_kfold_per_length_binders_train_old = per_allele_per_kfold_per_length_binders_train
per_allele_per_kfold_per_length_binders_test_old = per_allele_per_kfold_per_length_binders_test

if check_peptides:
    transform_dicts = None

per_allele_per_kfold_per_length_binders_train, per_allele_per_kfold_per_length_binders_test = merge_per_allele_sets_into_combination_sets(
    per_allele_per_kfold_per_length_binders_train, per_allele_per_kfold_per_length_binders_test,
    combination_size=combination_size,
    selected_alelles=ALLELES,
    total_splits=TOTAL_SPLITS,
    list_of_transform_dicts=None, position_component=params['position_component']
)
OLD_ALLELES = ALLELES
NEW_ALLELES = list(per_allele_per_kfold_per_length_binders_test.keys())
ALLELES = NEW_ALLELES

###############################
### Calculate weights
###############################


count_df_train = extract_count_df(per_allele_per_kfold_per_length_binders_train_old,
                                  selected_alleles=OLD_ALLELES,
                                  total_splits=TOTAL_SPLITS)
count_df_test = extract_count_df(per_allele_per_kfold_per_length_binders_test_old,
                                 selected_alleles=OLD_ALLELES,
                                 total_splits=TOTAL_SPLITS)

per_allele_per_kfold_per_length_weights = calculate_weights_based_on_length_counts(count_df_train,
                                                                                   selected_alleles=ALLELES,
                                                                                   total_splits=TOTAL_SPLITS,
                                                                                   selected_lengths=TARGET_LENGTHS,
                                                                                   use_weights=USE_WEIGHTS)


prepared_model = init_grouped_structure_model_joiners_and_cycles_here(
    states_per_group=int(params['states_per_group']),
    number_of_groups=params['groups'],
    CYCLE_PSEUDOCOUNT=params['cycle_pseudocount'],
    freeze_distr=params['freeze_distr'],
    initial_params=initial_params,
    start_cycle=params['start_cycle'],
    end_cycle=params['end_cycle'],
    self_cycles=params['self_cycle'],
    discrete_distribution=check_peptides,
    equal_probs_for_cycle=params['equal_probs_for_cycle'],
    tie_cycle_states=params['tie_cycle_states'],
    anchor_max_components=params['anchor_top_aas'],
    anchor_states=params['anchor_states'],
    model_complexity=params['model_complexity'],
    anchor_default_pseudocount= params['anchor_default_pseudocount'],
    tie_anchor_states=params['tie_anchor_states'],
                    intermediate_cycles=params['intermediate_cycles'],
                    equal_probs_for_intermediate_cycle=params['equal_probs_for_intermediate_cycle'],
                    tie_intermediate_cycles=params['tie_intermediate_cycles'],
                    freeze_int_cycle_distr=params['freeze_int_cycle_distr'],
    position_component=params['position_component']
)

#with open('model.png', 'w+') as f:
 #   prepared_model.plot(file=f, crop_zero=True, rotation=90, rotate=90)

#from hmm_visualization_methods import *
#plot_distributions_for_states(prepared_model,"test/", horizontal=False, discrete=check_peptides, initial_params=initial_params)

#make_pyviz_graph(prepared_model, "test/", precision=3)

###############################
### Actural HMM code
###############################

stop_threshold = 0
cutoff = 0.00005
per_symbol_portion = -5
for target_length in TARGET_LENGTHS:
    target_length_for_training = target_length
    train_size = len(per_allele_per_kfold_per_length_binders_train[ALLELES[0]][0][target_length])
    ## Stop threshold Hardcoded here
    stop_threshold += abs(cutoff * train_size * per_symbol_portion * target_length_for_training)
stop_threshold = stop_threshold / len(TARGET_LENGTHS)

params['stop_threshold'] = stop_threshold

####
# prepare models
####

per_allele_per_run_per_split_prepared_models = dict()
for allele_name in ALLELES:
    target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_').replace('/', '_')
    per_allele_per_run_per_split_prepared_models[allele_name] = dict()
    for run_num in range(NUM_RUNS):
        per_allele_per_run_per_split_prepared_models[allele_name][run_num] = dict()
        for split_num in range(TOTAL_SPLITS):
            # prepared_model  = init_grouped_structure_model(states_per_group=int(params['st']/params['groups']), number_of_groups=params['groups'])
            prepared_model = init_grouped_structure_model_joiners_and_cycles_here(
                states_per_group=int(params['states_per_group']),
                number_of_groups=params['groups'],
                CYCLE_PSEUDOCOUNT=params['cycle_pseudocount'],
                freeze_distr=params['freeze_distr'],
                initial_params=initial_params,
                start_cycle=params['start_cycle'],
                end_cycle=params['end_cycle'],
                self_cycles=params['self_cycle'],
                discrete_distribution=check_peptides,
                equal_probs_for_cycle=params['equal_probs_for_cycle'],
                tie_cycle_states=params['tie_cycle_states'],
                model_complexity=params['model_complexity'],
                anchor_default_pseudocount=params['anchor_default_pseudocount'],
                anchor_max_components=params['anchor_top_aas'],
                anchor_states=params['anchor_states'],
                tie_anchor_states=params['tie_anchor_states'],
                intermediate_cycles=params['intermediate_cycles'],
                equal_probs_for_intermediate_cycle=params['equal_probs_for_intermediate_cycle'],
                tie_intermediate_cycles=params['tie_intermediate_cycles'],
                freeze_int_cycle_distr=params['freeze_int_cycle_distr'],
                position_component=params['position_component']

            )
            prepared_model.name = f'{run_num}_run-{lengths_string}-{params_string}_model-{target_allele_name}-{split_num}'
            per_allele_per_run_per_split_prepared_models[allele_name][run_num][split_num] = prepared_model

####
# Do training
####
TOTAL_SPLITS = 1
per_name_models = dict()
per_allele_per_run_per_split_new_histories = dict()
for allele_name in ALLELES:
    per_allele_per_run_per_split_new_histories[allele_name] = dict()
    for run_num in range(NUM_RUNS):
        print(f"run {run_num}")
        per_kfold_per_length_data = per_allele_per_kfold_per_length_binders_train[allele_name]
        per_kfold_per_length_weights = per_allele_per_kfold_per_length_weights[allele_name]

        models, histories = train_model(allele_name=allele_name,
                                        per_kfold_per_length_data=per_kfold_per_length_data,
                                        per_kfold_per_length_weights_for_data=per_kfold_per_length_weights,
                                        target_lengths=TARGET_LENGTHS,
                                        multiple_check_input=params['multiple_check_input'],
                                        algorithm=params['alg'],
                                        verbose=params['verbose'],
                                        per_kfold_per_length_test_data=per_allele_per_kfold_per_length_binders_test[
                                            allele_name],
                                        stop_threshold=params['stop_threshold'],
                                        min_iterations=params['min_iters'],
                                        max_iterations=params['max_iters'],
                                        prepared_models=per_allele_per_run_per_split_prepared_models[allele_name][
                                            run_num],
                                        emission_pseudocount=params['emission_pseudocount'],
                                        transition_pseudocount=params['transition_pseudocount'],
                                        edge_inertia=params['edge_inertia'],
                                        distribution_inertia=params['distribution_inertia'],
                                        lr_decay=params['lr_decay'],
                                        use_pseudocount=params['use_pseudocount'],
                                        n_jobs=1)

        per_allele_per_run_per_split_prepared_models[allele_name][run_num] = models
        per_allele_per_run_per_split_new_histories[allele_name][run_num] = histories
        # per_allele_per_run_per_split_models[allele_name][run_num] = models
        for split_num, model in models.items():
            per_name_models[model.name] = model
        # save models
        for split_num, model in models.items():
            history = histories[split_num]
            save_model(TARGET_PATH_TO_RESULTS, model, history)

######
# After training
#####
# Sort by score
per_name_new_models = dict()
per_name_histories = dict()
for split_num in range(TOTAL_SPLITS):
    for allele_name in ALLELES:
        df = pd.DataFrame({"model": np.arange(NUM_RUNS),
                           "log_probability": [
                               per_allele_per_run_per_split_new_histories[allele_name][i][split_num].log_probabilities[
                                   -1]
                               for i in range(NUM_RUNS)],
                           # "test_log_probability":  [per_allele_per_run_per_split_new_histories[allele_name][i][split_num].test_log_probability[-1]
                           #                   for i in range(NUM_RUNS)]
                           })
        df = df.sort_values(by='log_probability', ascending=False).reset_index(drop=True)
        for new_index, model_run_num in zip(np.arange(len(df)), df.model):
            original_model = per_allele_per_run_per_split_prepared_models[allele_name][model_run_num][split_num]
            original_history = per_allele_per_run_per_split_new_histories[allele_name][model_run_num][split_num]
            original_model_name = original_model.name
            rest_part = original_model_name.split("_", 1)[1]
            new_name = f"{new_index}_{rest_part}"
            original_model.name = new_name
            per_name_new_models[new_name] = original_model
            per_name_histories[new_name] = original_history
per_name_models = per_name_new_models

# Save again
for name, model in per_name_models.items():
    history = per_name_histories[name]
    save_model(TARGET_PATH_TO_RESULTS, model, history)

#####
# save graph
#####


for i, model in enumerate(per_name_models.values()):
    print("Saving graph", i)
    # with open(f"{TARGET_PATH_TO_RESULTS}{model.name}/state_graph.png", 'w+') as f:
    #     model.plot(file=f, crop_zero=False)
    with open(f"{TARGET_PATH_TO_RESULTS}{model.name}/state_graph_cropped.png", 'w+') as f:
        model.plot(file=f, crop_zero=True)
### 
# save states
####
for i, model in enumerate(per_name_models.values()):
    print("saving  states for model", i)
    plot_distributions_for_states(model, TARGET_PATH_TO_RESULTS, horizontal=False, discrete=check_peptides,
                                  initial_params=initial_params)

###


###
# Coold pyviz vizualization
##

per_name_pyviz = dict()
for name, model in per_name_models.items():
    per_name_pyviz[name] = make_pyviz_graph(model, TARGET_PATH_TO_RESULTS, precision=3)
