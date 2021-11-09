import types

from pomegranate import *
import json
import numpy as np
import pandas as pd
from pomegranate.io import BatchedDataGenerator


def train_model(allele_name, per_kfold_per_length_data, target_lengths, name_prefix="", prepared_models=None,
                num_states=12, verbose=False, multiple_check_input=True, n_jobs=1, algorithm='baum-welch',
                lr_decay=0.0,
                minibatch_training=False,
                batches_per_epoch=None,
                batch_size=1, min_iterations=30,
                init_batch_size=None,
                emission_pseudocount=0.0,
                transition_pseudocount=0.0,
                use_pseudocount=False,
                edge_inertia=0.0,
                distribution_inertia=0.0
                ):

    per_split_model = dict()
    per_split_history = dict()

    assert type(prepared_models) in [dict, type(None)], "Prepared models should be either None (for new models) or per split models dict"

    for split_num, per_length_data in per_kfold_per_length_data.items():
        list_of_arrays = [per_length_data[target_length] for target_length in target_lengths]
        binders_array = np.concatenate(list_of_arrays)
        print("split {}".format(split_num))
        print("train size {}".format(len(binders_array)))
        print("Shuffling for fairness ")
        rng = np.random.default_rng()
        rng.shuffle(binders_array)

        sample_X = np.array([[char for char in item] for item in binders_array], dtype=object)
        if minibatch_training:
            sequence_generator = BatchedDataGenerator(sample_X, batches_per_epoch=batches_per_epoch, batch_size=batch_size)
            total_batches = int(len(sequence_generator) / sequence_generator.batch_size)
            if init_batch_size is None:
                if batches_per_epoch is None:
                    init_batch_size = total_batches
                else:
                    assert batches_per_epoch < total_batches, f"Batches per epoch should be {total_batches} or lower for given batch size of {batch_size}"
                    init_batch_size = batches_per_epoch
            print(f"Mini_batch: Train Length {len(sequence_generator)}, "
                  f"batch_size {sequence_generator.batch_size}, batches_per_epoch {sequence_generator.batches_per_epoch}, "
                  f"init_batch_size {init_batch_size}")
        else:
            sequence_generator = sample_X
            print(f"Simple training: Train Length {len(sequence_generator)}, "
                  f"init_batch_size {init_batch_size}")
            if init_batch_size is not None:
                assert init_batch_size < len(sequence_generator), f"init batch size should be {len(sequence_generator)} or lower "\
                                                                  f"for given batch size of {batch_size}"


    # init_batch_size = max(int(batches_per_epoch / 10), 1) if batches_per_epoch is not None else int(len(sequence_generator)/10)
            # init_batch_size = 1 if batch_size is None and batches_per_epoch is None else init_batch_size

        if prepared_models is None:
            lengths_string = "-".join(map(str, target_lengths))
            name_postfix = name_prefix + "-" + lengths_string
            state_names = [f"s{i}_{lengths_string}" for i in range(num_states)]
            target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
            # fill model name
            params = {
                "st": num_states,
                "alg": algorithm,
                #"batches": batches_per_epoch,
                #"batch_size": batch_size,
                #"init_batch_size": init_batch_size,
                #"lr": lr_decay,
                #"edge_inertia": edge_inertia,
               # "distribution_inertia": distribution_inertia,
                #"emission_pseudocount": emission_pseudocount,
               # "transition_pseudocount": transition_pseudocount,
                "min_iters": min_iterations,
               # "use_pseudocount": use_pseudocount
            }
            params_string = "-".join([f"{value}-{key}" for key, value in params.items()])
            model_name = f"{name_postfix}_model"
            model, history = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution,
                                                            n_components=num_states,
                                                            X=sequence_generator,
                                                            end_state=True,
                                                            # max_iterations=100,
                                                            stop_threshold=5e-5,
                                                            # batches_per_epoch=10,
                                                            name=model_name,
                                                            state_names=state_names,
                                                            return_history=True,
                                                            verbose=verbose,
                                                            multiple_check_input=multiple_check_input,
                                                            lr_decay=lr_decay,
                                                            edge_inertia=edge_inertia,
                                                            distribution_inertia=distribution_inertia,
                                                            min_iterations=min_iterations,
                                                            initialization_batch_size=init_batch_size,
                                                            emission_pseudocount=emission_pseudocount,
                                                            # init='random' # 'first-k', 'random', 'kmeans++', or 'kmeans||' Only for non-discrete discributions
                                                            transition_pseudocount=transition_pseudocount,
                                                            use_pseudocount= use_pseudocount,
                                                            # distribution_inertia
                                                            # edge_inertia
                                                            # inertia
                                                            n_jobs=n_jobs,
                                                            algorithm=algorithm
                                                            #labels
                                                            #weights  An array of weights, one for each sequence to train on. If None, all sequences are equally weighted
                                                            )
            model_name = f"{name_postfix}-{params_string}_model-{target_allele_name}-{split_num}"
            model.name = model_name
        else:
            model = prepared_models[split_num]
            model, history = model.fit(sample_X,
                                       stop_threshold=5e-5,
                                       return_history=True,
                                       verbose=verbose,
                                       multiple_check_input=multiple_check_input,
                                       n_jobs=n_jobs,
                                       algorithm=algorithm,
                                       lr_decay=lr_decay,
                                       distribution_inertia=distribution_inertia,
                                       edge_inertia=edge_inertia,
                                       batches_per_epoch=batches_per_epoch
                                       )
        per_split_model[split_num] = model
        per_split_history[split_num] = history
    return per_split_model, per_split_history


# Add states to existing model
def add_more_states_and_reset_transitions(source_model, ADD_STATES_NUM, states_prefix,  new_name, new_probs="fair"):

    old_start_index = source_model.start_index
    old_end_index = source_model.end_index
    old_states = source_model.states.copy()
    old_state_name_to_index = {state.name : i for i, state in enumerate(old_states) }
    old_states_names = {state.name for state in old_states}
    old_transitions = source_model.dense_transition_matrix()
    state_distributions = []
    model = source_model.copy()

    print(f"Initial length {len(model.states)}")
    keymap = model.keymap[0]
    for i in range(ADD_STATES_NUM):
        d = DiscreteDistribution({key: 1. / len(keymap) for key in keymap})
        state_distributions.append(d)
    state_names = [f"{states_prefix}_{state_num}" for state_num in range(ADD_STATES_NUM)]
    states_to_add = [State(distribution, name=name) for name, distribution in zip(state_names, state_distributions)]
    print(f"num states to add {len(states_to_add)}")
    model.add_states(states_to_add)
    model.name = new_name
    model.start.name = f"{new_name}-start"
    model.end.name = f"{new_name}-end"
    model.bake(verbose=True, merge=None) # merge None because we don't want to delete the states without nodes
    print(f"After length {len(model.states)}")

    # total states after adding
    states = model.states
    states.remove(model.start)
    states.remove(model.end)
    n = len(states)
    print(f" N is  {n}")

    # Connect the start of the model to the appropriate state
    sum_from_start = 0
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[old_start_index, old_state_name_to_index[state.name]]
        else:
            if new_probs == "fair":
                prob = 1 / ADD_STATES_NUM  # we want to have 0.5 probability in final normalized version for new states
            elif new_probs == "tiny":
                prob = 0.05
            else:
                raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
        if prob != 0:
            # print(f"ADD transition from {model.start.name} {state.name} with prob {prob}")
            model.add_transition(model.start, state, prob)
            sum_from_start += prob
    print(f"sum from start {sum_from_start}")

    # Connect all states to each other if they have a non-zero probability
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if from_state.name in old_states_names and to_state.name in old_states_names:
                prob = old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
            elif from_state.name in old_states_names:
                if new_probs == "fair":
                    prob = 1 / ADD_STATES_NUM  # probabilities from old to new will be 1/(num_new)
                elif new_probs == "tiny":
                    prob = 0.05
                else:
                    raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
            else: # from new state
                prob = 1.0 # if from new state  equal probs from all the states + end state
            if prob != 0:
                # print(f"ADD transition from {from_state.name} {to_state.name} with prob {prob}")
                model.add_transition(from_state, to_state, prob)

    # Connect states to the end of the model if a non-zero probability
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[ old_state_name_to_index[state.name], old_end_index]
        else:
            prob = 1.0  # if from new state  equal probs from all the states + end state
        if prob != 0:
            #print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            model.add_transition(state, model.end, prob)
    model.bake(verbose=True)
    return model


def save_model(model, history, TARGET_PATH_TO_RESULTS):
    json_object = model.to_json()
    MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{model.name}/"
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    with open(f"{MODEL_FOLDER}/model.json", 'w+') as f:
        json.dump(json_object, f)
    history_df = pd.DataFrame()
    history_df['total_improvement'] = history.total_improvement
    history_df['improvements'] = history.improvements
    history_df['log_probabilities'] = history.log_probabilities
    history_df['epoch_start_times'] = history.epoch_start_times
    history_df['epoch_end_times'] = history.epoch_end_times
    history_df['epoch_durations'] = history.epoch_durations
    history_df['epochs'] = history.epochs
    history_df['learning_rates'] = history.learning_rates
    history_df['n_seen_batches'] = history.n_seen_batches
    history_df.to_csv(f'{MODEL_FOLDER}/history.csv', index=False)


def load_model(TARGET_PATH_TO_RESULTS, model_name):
    MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{model_name}/"
    with open(f"{MODEL_FOLDER}/model.json") as f:
        json_object = json.load(f)
        model = HiddenMarkovModel.from_json(json_object)
    return model
