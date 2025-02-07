
import sys
PATH_TO_PREDICTOR_HOME = "../.."
sys.path.append(PATH_TO_PREDICTOR_HOME)

import pomegranate.io
from pomegranate import *
import training_parameters
from peptides_utils import defineClass, defineOrganism, make_logo_for_data
from data_reading_methods import remove_unused_lengths, transform_data_to_properties_and_join_alleles, calculate_weights_based_on_length_counts
from data_reading_methods import get_available_alleles, get_train_test_data, join_dicts

import random

random.seed(10)

from hmm_logic_methods import train_model_prepared,  \
    train_model_batched, save_model, \
    add_more_states_and_reset_transitions, \
    build_model_based_on_params, \
    prepare_multiple_models, train_multiple_models, \
reorder_models_by_score_and_flatten_to_by_name_list, hierarchically_train_splited_models
from hmm_visualization_methods import *



METHOD = "hmm_pomegranate"

mixture_num = int(sys.argv[1])

###############################
### Read data
###############################
hmm_params = training_parameters.ExperimentParams(experiment_name="split_mixmhc")
hmm_params.model_training_params = training_parameters.SimpleModelClassIIParamsMixMHCsmall()
hmm_params.data_scenario_params = training_parameters.MixMHCpredDataParams()
hmm_params.data_scenario_params.mixmhc_mixture_name = hmm_params.data_scenario_params.extra['mixed_samples'][mixture_num]
model_training_params = hmm_params.model_training_params
print(f"Mixture will be {hmm_params.data_scenario_params.extra['mixed_samples'][mixture_num]}")


per_allele_per_kfold_per_length_binders_train, \
per_allele_per_kfold_per_length_binders_test, additional_data = get_train_test_data(hmm_params)

PARSED_ALLELES = list(per_allele_per_kfold_per_length_binders_train.keys())
print(sorted(PARSED_ALLELES))

model_training_params.lengths_to_use = list(set.intersection(set(model_training_params.lengths_to_use), set(per_allele_per_kfold_per_length_binders_train[PARSED_ALLELES[0]][0].keys())))

model_training_params: training_parameters.ModelTrainingParams = hmm_params.model_training_params
#model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in ['HLA-DRB1*04:01']]
current_mix = PARSED_ALLELES[0]
model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in [current_mix]]
#model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in ['decoy-allele-78']]
# remove unused lengths (used are specified in ModelTrainingParams.model_training_params
per_allele_per_kfold_per_length_binders_train = remove_unused_lengths(per_allele_per_kfold_per_length_binders_train, experiment_params=hmm_params)
per_allele_per_kfold_per_length_binders_test = remove_unused_lengths(per_allele_per_kfold_per_length_binders_test,experiment_params=hmm_params)

# transform to properties if needed and join multiple alleles
train_data, test_data, NEW_ALLELES, \
old_train_data, old_test_data, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    hmm_params.model_training_params
)
# Calculate weights for the training data based on peptide couns/lengths for unmerged data
train_data_weigths = calculate_weights_based_on_length_counts(old_train_data, experiment_params=hmm_params)




###
# prepare and train models models
####
#prepare models
per_allele_per_run_per_split_prepared_models = prepare_multiple_models(hmm_params)
#do training

result_models, result_histories =  train_multiple_models(
    prepared_models=per_allele_per_run_per_split_prepared_models,
    train_data=train_data,
    test_data=test_data,
    train_data_weigths=train_data_weigths,
    experiment_params=hmm_params,
    subfolder_to_safe_result="root")


per_name_models, per_name_histories, original_alelle_sortings = reorder_models_by_score_and_flatten_to_by_name_list(histories=result_histories,
                                                                models=result_models, experiment_params=hmm_params,subfolder_to_safe_result="root" )
save_all_visualization_results(per_name_models, per_name_histories, experiment_params=hmm_params, subfolder_to_safe_result="root", subset=40)



####
#  check how models align the data
####

import importlib
import hmm_logic_methods
import hmm_visualization_methods
import logomaker
from peptides_utils import get_frequencies
hmm_logic_methods= importlib.reload(hmm_logic_methods)
hmm_logic_methods= importlib.reload(hmm_visualization_methods)
from hmm_logic_methods import check_model_alignment, init_distribution
allele_data_df = join_dicts(train_data)[PARSED_ALLELES[0]]
allele_name = PARSED_ALLELES[0].replace('-', '_').replace('*', '_').replace(':', '_')

info_matrix, _, fig = check_model_alignment(per_name_models[list(per_name_models.keys())[1]], allele_data_df, allele_name)
filename=f"{allele_name}-split-root-example_of_alignment.svg"
path_to_save_pictures = f"{hmm_params.experiment_result_data_path}"
save_figure_to_svg(fig, dir=path_to_save_pictures, filename=f"{filename}.svg")

## Average between models
from hmm_logic_methods import get_information_content_estimation_for_models
statistics_estimate_models_fraction = model_training_params.statistics_estimate_model_fraction

name_to_info_value, allele_to_info_matrix, allele_to_info_value = get_information_content_estimation_for_models(per_name_models, models_fraction=statistics_estimate_models_fraction)
model_name_to_allele = {
    model_name : f"{model_name.split('-')[-2]}" for model_name in per_name_models.keys()
}
path_to_save_pictures = f"{hmm_params.experiment_result_data_path}"
info_threshold = model_training_params.trash_model_information_threshold
passed_filter = {key for key, info_value in name_to_info_value.items() if info_value > info_threshold}
print(f"For root {len(passed_filter)} models passed the info filter out of {len(per_name_models)}")
for allele_name in allele_to_info_matrix.keys():
    info_value =allele_to_info_value[allele_name]
    passed = info_value > info_threshold
    main_title = f"Allele {allele_name} \n  info ({np.round(info_value,2)}) passed ({passed})"
    filename=f"{allele_name}-split-root.svg"
    fig, ax = plt.subplots(ncols      =1, nrows=1, figsize=(5, 3))
    ww_logo = logomaker.Logo(allele_to_info_matrix[allele_name], font_name='Liberation Sans Narrow', flip_below=True,
                             ax=ax)
    ww_logo.ax.set_title(f"branch")
    fig.suptitle(main_title)
    plt.tight_layout()
    save_figure_to_svg(fig, dir=path_to_save_pictures, filename=f"{filename}.svg")
    plt.close(fig)


#####
#  try to do splits
####

import sys
#orig_std = sys.stdout
# sys.stdout = f
models, split_pictures = hierarchically_train_splited_models(per_name_models=per_name_models,
                                    train_data=train_data,
                                    test_data=test_data,
                                    additional_data=additional_data,
                                    experiment_params=hmm_params,
                                    total_layers_to_do=3)
