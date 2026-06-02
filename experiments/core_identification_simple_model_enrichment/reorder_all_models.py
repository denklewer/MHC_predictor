import os
from hmm_logic_methods import  save_model, load_model
# Read all models


base_dir = "experiment_results"
base_models = [f"base_model_{i}" for i in range(4,10)]
new_models = [f"second_model_{i}" for i in range(4,9)]
experiments = base_models + new_models


def read_all_models(models_dir):
    parsed_alleles = os.listdir(models_dir)
    per_allele_per_split_per_run_model = dict()
    per_allele_per_split_per_run_history = dict()
    for parsed_allele in parsed_alleles:
        per_allele_per_split_per_run_model[parsed_allele] = dict()
        allele_dir = f"{models_dir}/{parsed_allele}"
        for split_string in os.listdir(allele_dir):
            per_allele_per_split_per_run_model[parsed_allele][split_string] = dict()
            per_allele_per_split_per_run_history[parsed_allele][split_string] = dict()
            split_dir = f"{allele_dir}/{split_string}"
            for run_string in os.listdir(split_dir):
                run_dir = f"{split_dir}/{run_string}/"
                run_index = run_string.split("_")[0]
                model = load_model(run_dir, run_dir)
                history = pd.read_csv(os.path.join(run_dir, "history.csv"))
                per_allele_per_split_per_run_model[parsed_allele][run_index] = model
                per_allele_per_split_per_run_history[parsed_allele][run_index] = history
    return per_allele_per_split_per_run_model, per_allele_per_split_per_run_history



def reorder_per_run_models_by_score(run_models, run_histories, path):
    # Create a DataFrame with models and their log probabilities
    df = pd.DataFrame({
        "model": list(run_models.keys()),
        "log_probability": [run_histories[run].log_probabilities[-1] for run in run_models.keys()]
    })

    # Sort the DataFrame by 'log_probability'
    sorted_df = df.sort_values(by='log_probability', ascending=False).reset_index(drop=True)
    # Sort original keys
    original_keys_sorted =list(sorted(run_models.keys()))
    # Create a dictionary to map "run new" to "run old" (the original order)
    run_mapping = {sorted_df['model'].iloc[i]: original_keys_sorted[i] for i in range(len(sorted_df))}
    sorting_log = [f"{key} -> {item}" for key, item in run_models.items()]
    sorted_histories = {f"run[{i:04d}]": run_histories[sorted_df['model'].iloc[i]] for i in range(len(sorted_df))}

    # Generate per_run_models with formatted run indices
    per_run_models = {f"run[{i:04d}]": run_models[sorted_df['model'].iloc[i]] for i in range(len(sorted_df))}

    for run_ind, model in per_run_models.items():
        original_model_name = model.name
        rest_part = "_".join(original_model_name.split("_")[1:])
        # print(rest_part)
        print(f"Run {run_ind}", end=" ")
        # Update the model name with the new formatted index
        # new_name = f"run[{run_ind:04d}_{rest_part}"
        new_name = f"{run_ind}_{rest_part}"
        model.name = new_name

        # Get the original run key from sorted_df
        # path_run = path + f'//{run_ind}'
        #run_key = sorted_df['model'].iloc[int(run_ind[4:-1])]  # Extract the numeric part from 'run[0000]'
        save_model(path + '//', model, run_histories[run_ind])

    return per_run_models, sorted_histories, run_mapping




def reorder_models_by_run_score_for_splits(histories, models, experiment_path, subfolder_to_safe_result):
    path_to_save_models = os.path.join(experiment_path, subfolder_to_safe_result)
    per_allele_new_models = {}
    per_split_allele_original_sortings = {}
    per_split_run_mapping = {}
    per_allele_histories = {}

    for allele_name, splits in models.items():
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_').replace('/', '_')
        path_to_save_models_allele = path_to_save_models + f"//{target_allele_name}"
        per_allele_new_models[allele_name] = {}
        per_split_allele_original_sortings[allele_name] = {}
        per_split_run_mapping[allele_name] = {}
        per_allele_histories[allele_name] = {}
        print("Allele: ", allele_name)

        for split_index, per_run_models in splits.items():
            print("Split: ", split_index, end=" ")
            path_to_save_models_split = path_to_save_models_allele + f'//{split_index}'
            sorted_model_per_run, sorted_histories_per_run, run_mapping = reorder_per_run_models_by_score(
                per_run_models, histories[allele_name][split_index], path_to_save_models_split)
            per_allele_new_models[allele_name][split_index] = sorted_model_per_run
            per_split_run_mapping[allele_name][split_index] = run_mapping
            per_allele_histories[allele_name][split_index] = sorted_histories_per_run

    return per_allele_new_models, per_allele_histories, per_split_run_mapping




for experiment in experiments:
    # read binders
    binders_subdir = f"{base_dir}/binders/"
    parsed_alleles = os.listdir(binders_subdir)
    per_allele_per_split_per_run_model, per_allele_per_split_per_run_history = read_all_models(binders_subdir)
    per_allele_per_split_sorted_runs_b, per_allele_per_split_sorted_hist_b, per_split_run_mapping_b = reorder_models_by_run_score_for_splits(
        per_allele_per_split_per_run_history, per_allele_per_split_per_run_model, binders_subdir, subfolder_to_safe_result="reordered")
    # read nonbinders
    nonbinders_subdir = f"{base_dir}/nonbinders/"
    parsed_alleles = os.listdir(binders_subdir)
    per_allele_per_split_per_run_model, per_allele_per_split_per_run_history = read_all_models(binders_subdir)
    per_allele_per_split_sorted_runs_b, per_allele_per_split_sorted_hist_b, per_split_run_mapping_b = reorder_models_by_run_score_for_splits(
        per_allele_per_split_per_run_history, per_allele_per_split_per_run_model, binders_subdir, subfolder_to_safe_result="reordered")





