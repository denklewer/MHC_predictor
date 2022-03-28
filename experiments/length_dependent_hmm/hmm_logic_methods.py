import types
from pomegranate import *
import json
import numpy as np
import pandas as pd
from pomegranate.io import BatchedDataGenerator, SequenceGenerator
from hmm_visualization_methods import convert_graph_to_good_format
import networkx as nx
from tqdm import tqdm


def train_model(allele_name, per_kfold_per_length_data, target_lengths, name_prefix="",
                per_kfold_per_length_test_data=None,
                per_kfold_per_length_weights_for_data=None,
                prepared_models=None,
                num_states=12, verbose=False, multiple_check_input=True, n_jobs=1, algorithm='baum-welch',
                lr_decay=0.0,
                minibatch_training=False,
                batches_per_epoch=None,
                batch_size=1, min_iterations=30, max_iterations=1e8,
                init_batch_size=None,
                emission_pseudocount=0.0,
                transition_pseudocount=0.0,
                use_pseudocount=False,
                edge_inertia=0.0,
                distribution_inertia=0.0,
                stop_threshold=5e-5
                ):

    per_split_model = dict()
    per_split_history = dict()

    assert type(prepared_models) in [dict, type(None)], "Prepared models should be either None (for new models) or per split models dict"

    for split_num, per_length_data in per_kfold_per_length_data.items():
        per_length_current_weights = per_kfold_per_length_weights_for_data[split_num]
        per_length_test_data = per_kfold_per_length_test_data[split_num]
        per_length_counts = {target_length: len(per_length_data[target_length]) for target_length in target_lengths}

        list_of_arrays = [per_length_data[target_length] for target_length in target_lengths]
        list_of_arrays_for_weights = [np.array(per_length_current_weights[target_length]) for target_length in target_lengths]
        list_of_test_arrays = [per_length_test_data[target_length] for target_length in target_lengths]
        binders_array = np.concatenate(list_of_arrays)
        weights_array = np.concatenate(list_of_arrays_for_weights)


        binders_test_array = np.concatenate(list_of_test_arrays)
        print("split {}".format(split_num))
        print("train size {}".format(len(binders_array)))
        # total_length = len(binders_array)
        #  weights = [1 - per_length_counts[len(peptide)]/total_length for peptide in binders_array]
        # print("weights for lengths {}".format({key: 1 - value/total_length for key, value in per_length_counts.items()}))
        print("Shuffling for fairness ")

        rng = np.random.default_rng()
        # shuffle train data and weights in unison
        new_indexes = rng.permutation(len(binders_array))
        binders_array = binders_array[new_indexes]
        weights_array = weights_array[new_indexes]

        # just shuffle test data
        rng.shuffle(binders_test_array)
        sample_X = np.array([[char for char in item] for item in binders_array], dtype=object)
        sample_X_test =  np.array([[char for char in item] for item in binders_test_array], dtype=object)
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
            sequence_test_generator = sample_X_test
        else:
            sequence_generator = sample_X
            sequence_test_generator = SequenceGenerator(sample_X_test)
            print(f"Simple training: Train Length {len(sequence_generator)}, "
                  f"init_batch_size {init_batch_size}")
            if init_batch_size is not None:
                assert init_batch_size < len(sequence_generator), f"init batch size should be {len(sequence_generator)} or lower "\



    # init_batch_size = max(int(batches_per_epoch / 10), 1) if batches_per_epoch is not None else int(len(sequence_generator)/10)
            # init_batch_size = 1 if batch_size is None and batches_per_epoch is None else init_batch_size

        if prepared_models is None:
            lengths_string = "-".join(map(str, target_lengths))
            name_postfix = name_prefix + "-" + lengths_string
            state_names = ["s{:03d}".format(i) for i in range(num_states)]
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
                "min_iters": min_iterations
               # "use_pseudocount": use_pseudocount
            }
            if max_iterations != 1e8:
                params["max_iters"] = max_iterations
            params_string = "-".join([f"{value}-{key}" for key, value in params.items()])
            model_name = f"{name_postfix}_model"
            model, history = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution,
                                                            n_components=num_states,
                                                            X=sequence_generator,
                                                            X_test=sequence_test_generator,
                                                            end_state=True,
                                                            # max_iterations=100,
                                                            stop_threshold=stop_threshold,
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
                                                            max_iterations=max_iterations,
                                                            initialization_batch_size=init_batch_size,
                                                            emission_pseudocount=emission_pseudocount,
                                                            # init='random' # 'first-k', 'random', 'kmeans++', or 'kmeans||' Only for non-discrete discributions
                                                            transition_pseudocount=transition_pseudocount,
                                                            use_pseudocount= use_pseudocount,
                                                            # inertia
                                                            n_jobs=n_jobs,
                                                            algorithm=algorithm,
                                                            weights = weights_array
                                                            #labels
                                                            #weights  An array of weights, one for each sequence to train on. If None, all sequences are equally weighted
                                                            )
            model_name = f"{name_postfix}-{params_string}_model-{target_allele_name}-{split_num}"
            model.name = model_name
        else:
            model = prepared_models[split_num]
            model, history = model.fit(sample_X,
                                       sequences_test =sequence_test_generator,
                                       stop_threshold=stop_threshold,
                                       return_history=True,
                                       verbose=verbose,
                                       multiple_check_input=multiple_check_input,
                                       n_jobs=n_jobs,
                                       algorithm=algorithm,
                                       lr_decay=lr_decay,
                                       distribution_inertia=distribution_inertia,
                                       edge_inertia=edge_inertia,
                                       batches_per_epoch=batches_per_epoch,
                                       min_iterations= min_iterations,
                                       emission_pseudocount=emission_pseudocount,
                                       transition_pseudocount=transition_pseudocount,
                                       use_pseudocount= use_pseudocount,
                                       weights=weights_array
                                       )
        per_split_model[split_num] = model
        per_split_model[split_num] = model
        per_split_history[split_num] = history
        break
    return per_split_model, per_split_history


# Add states to existing model
def add_more_states_and_reset_transitions(source_model, ADD_STATES_NUM, new_sates_names=[],  new_name="new_model", new_probs="fair"):

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
        d = DiscreteDistribution({key: prob for key, prob in zip(keymap,np.random.uniform(0, 1, size=len(keymap)))})
        state_distributions.append(d)

    state_names = new_sates_names
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

    from_start_probs = list()
    for to_state in states:
        if to_state.name in old_states_names:
            from_start_probs.append(old_transitions[old_start_index, old_state_name_to_index[to_state.name]])

    median_from_start_prob = np.mean(from_start_probs)

    # start probabilities
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[old_start_index, old_state_name_to_index[state.name]]
        else:
            if new_probs == "fair":
                prob = median_from_start_prob # we want to have fair prob(median)
                # prob = 1 / ADD_STATES_NUM  # we want to have 0.5 probability in final normalized version for new states
            elif new_probs == "tiny":
                prob = 0.04
            else:
                raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
        if prob != 0:
            # print(f"ADD transition from {model.start.name} {state.name} with prob {prob}")
            model.add_transition(model.start, state, prob)
            sum_from_start += prob
    print(f"sum from start {sum_from_start}")

    # Connect all states to each other if they have a non-zero probability
    for i, from_state in enumerate(states):

        # calculate from probs
        from_probs = list()
        median_from_prob = 0
        for to_state in states:
            if from_state.name in old_states_names and to_state.name in old_states_names:
                from_probs.append(old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]])
        if from_state.name in old_states_names:
            median_from_prob = np.mean(from_probs)

        for j, to_state in enumerate(states):
            if from_state.name in old_states_names and to_state.name in old_states_names:
                prob = old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
            elif from_state.name in old_states_names:
                if new_probs == "fair":
                    prob = median_from_prob # probabilities from old to new will be median of all from probabilities
                    # prob = 1 / ADD_STATES_NUM  # probabilities from old to new will be 1/(num_new)
                elif new_probs == "tiny":
                    prob = 0.04
                else:
                    raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
            else: # from new state
                if to_state.name in old_states_names:
                    prob = 1.0 # to old state prob is 1
                else:
                    prob = 0 # from new to old prob is 0
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
    model.bake(verbose=False)
    return model

def init_grouped_structure_model(states_per_group = 5, number_of_groups=5, amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    START_WEIGHT = 1
    INTRAGROUP_WEIGHT = 4
    INTERGROUP_WEIGHT = 1
    END_WEIGHT = 4
    amino_acids_list = list(amino_acids)
    distributions = list()

    for group_number in range(number_of_groups):
        for state_num in range(states_per_group):
            d = DiscreteDistribution({key: prob for key, prob in zip(amino_acids_list,np.random.uniform(0, 1, size=len(amino_acids_list)))})
            distributions.append(d)

    num_states = states_per_group * number_of_groups
    state_names = ["s{:03d}".format(i) for i in range(num_states)]

    transition_matrix = numpy.zeros((num_states, num_states))


    for group_number in range(number_of_groups):
        for i in range(states_per_group):
            # fill transitions inside each group with 3

            total_rest_portion = INTRAGROUP_WEIGHT*states_per_group
            for j in range(states_per_group):
                state_num_from = group_number * states_per_group + i
                state_num_to = group_number * states_per_group + j
                if state_num_from != state_num_to:
                    transition_matrix[state_num_from][state_num_to] = INTRAGROUP_WEIGHT

            # transition to next group is 1
            if group_number != number_of_groups - 1:
                for j in range(states_per_group):
                    state_num_from = group_number * states_per_group + i
                    state_num_to = (group_number + 1) * states_per_group + j
                    transition_matrix[state_num_from][state_num_to] = INTERGROUP_WEIGHT
    model_name = 'grouped_model'

    # start only for first group
    start_probabilities = [START_WEIGHT if i < states_per_group else 0 for i in range(num_states)]
    # do not restrict ending in multiple states, so probs are equal to transitions inside
    end_probabilities = [0 if i < states_per_group * (number_of_groups - 1) else INTRAGROUP_WEIGHT for i in range(num_states)]
    assert sum(start_probabilities) == states_per_group * START_WEIGHT
    assert sum(end_probabilities) == states_per_group * END_WEIGHT

    model =HiddenMarkovModel.from_matrix(
        name=model_name,
        transition_probabilities=transition_matrix,
        starts=start_probabilities,
        distributions=distributions,
        ends=end_probabilities,
        state_names=state_names,
        verbose=True)

    model.bake(verbose=True)
    return model






# Add states to existing model
def add_more_states_to_the_end(source_model, ADD_STATES_NUM, new_sates_names=[],  new_name="new_model", new_probs="fair", edge_pseudocount=None):

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
        d = DiscreteDistribution({key: prob for key, prob in zip(keymap,np.random.uniform(0, 1, size=len(keymap)))})
        state_distributions.append(d)

    state_names = new_sates_names
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
            model.add_transition(model.start, state, prob, pseudocount=edge_pseudocount)
            sum_from_start += prob
    print(f"sum from start {sum_from_start}")

    # Connect all states to each other if they have a non-zero probability
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if from_state.name in old_states_names and to_state.name in old_states_names:
                if to_state.name != model.end.name:
                    prob = old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
                # else:
                #
                #     prob = 0
            elif from_state.name in old_states_names:
                if new_probs == "fair":
                    prob = 1 / ADD_STATES_NUM  # probabilities from old to new will be 1/(num_new)
                elif new_probs == "tiny":
                    prob = 0.05
                else:
                    raise RuntimeError("new_probs argument shoul be 'fair' or  'tiny'")
            else: # from new state
                prob = 0
            if prob != 0:
                # print(f"ADD transition from {from_state.name} {to_state.name} with prob {prob}")
                model.add_transition(from_state, to_state, prob, pseudocount=edge_pseudocount)

    # Connect states to the end of the model if a non-zero probability
    for state in states:
        if state.name in old_states_names:
            prob = 0
        else:
            prob = 1.0  # if from new state  equal probs from all the states + end state
        if prob != 0:
            #print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            model.add_transition(state, model.end, prob, pseudocount=edge_pseudocount)
    model.bake(verbose=False)
    return model


def copy_model_with_pseudocounts(source_model, precision = 4, edge_pseudocount=None):
    new_model = source_model.copy()

    transitions = new_model.dense_transition_matrix()
    # total states after adding
    states = new_model.states
    state_name_to_index = {state.name : i for i, state in enumerate(states) }
    states_names = {state.name for state in states}
    start_index = new_model.start_index
    end_index = new_model.end_index
    states.remove(new_model.start)
    states.remove(new_model.end)
    n = len(states)
    print(f" N is  {n}")
    # Connect the start of the model to the appropriate state
    sum_from_start = 0
    for state in states:
        prob = transitions[start_index, state_name_to_index[state.name]]
        prob = round(prob, precision)
        if prob != 0:
            # print(f"ADD transition from {model.start.name} {state.name} with prob {prob}")
            new_model.add_transition(new_model.start, state, prob, pseudocount=edge_pseudocount)
            sum_from_start += prob
        else:
            new_model.add_transition(new_model.start, state, prob, pseudocount=0)

    print(f"sum from start {sum_from_start}")

    # Connect all states to each other if they have a non-zero probability
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = 0
            if to_state.name != new_model.end.name:
                prob = transitions[state_name_to_index[from_state.name], state_name_to_index[to_state.name]]
                prob = round(prob, precision)

            if prob != 0:
                # print(f"ADD transition from {from_state.name} {to_state.name} with prob {prob}")
                new_model.add_transition(from_state, to_state, prob, pseudocount=edge_pseudocount)
            else:
                new_model.add_transition(from_state, to_state, prob, pseudocount=0)
    # Connect states to the end of the model if a non-zero probability
    for state in states:
        prob = transitions[state_name_to_index[state.name], end_index]
        prob = round(prob, precision)
        if prob != 0:
            #print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            new_model.add_transition(state, new_model.end, prob, pseudocount=edge_pseudocount)
        else:
            new_model.add_transition(state, new_model.end, prob, pseudocount=0)
    new_model.bake(verbose=False)
    return new_model

def create_model_with_separated_amino_acids(target_length=9):
    groups = {
        0: ['G', 'S'],
        1: ['T', 'Y', 'C'],
        2: ['Q', 'N'],
        3: ['K', 'R', 'H'],
        4: ['D', 'E'],
        5: ['A', 'V', 'L', 'I'],
        6: ['P', 'W', 'F', 'M'],
    }

    keymap = {'H': 0,
               'R': 1,
               'V': 2,
               'S': 3,
               'N': 4,
               'Q': 5,
               'W': 6,
               'K': 7,
               'M': 8,
               'D': 9,
               'I': 10,
               'Y': 11,
               'C': 12,
               'L': 13,
               'G': 14,
               'F': 15,
               'A': 16,
               'P': 17,
               'T': 18,
               'E': 19}

    distributions = []

    num_states = len(groups)*target_length
    transition_matrix = numpy.zeros((num_states, num_states))

    for state_position in range(target_length):

        for group_number in range(len(groups)):
            group_aas = groups[group_number]
            distr_probs = {aa: 0 for aa in keymap}
            non_zero_probs = np.random.uniform(0, 1, size=len(group_aas))
            for aa, prob in zip(group_aas, non_zero_probs):
                distr_probs[aa] = prob
            distributions.append(DiscreteDistribution(distr_probs))
        if state_position > 0:
            current_indecies = [i for i in range(state_position * len(groups), (state_position + 1) * len(groups))]
            previous_indecies = [i for i in range((state_position-1) * len(groups), state_position * len(groups))]
            for i in previous_indecies:
                for j in current_indecies:
                    transition_matrix[i][j] = 1/len(groups)

    model_name = 'splited_model'
    state_names = ["s{:03d}".format(i) for i in range(num_states)]


    # transition_matrix = numpy.ones((num_states, num_states)) / num_states
    start_probabilities = [1/len(groups) if i < len(groups) else 0 for i in range(num_states)]
    end_probabilities = [1 if i >= num_states-len(groups) else 0 for i in range(num_states)]
    # numpy.ones(num_states) / num_states
    # end_probabilities = numpy.ones(num_states) / num_states

    model =HiddenMarkovModel.from_matrix(
        name=model_name,
        transition_probabilities=transition_matrix,
        starts=start_probabilities,
        distributions=distributions,
        ends=end_probabilities,
        state_names=state_names,
        verbose=True)
    return model



def save_model(TARGET_PATH_TO_RESULTS, model, history=None ):
    json_object = model.to_json()
    MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{model.name}/"
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    with open(f"{MODEL_FOLDER}/model.json", 'w+') as f:
        json.dump(json_object, f)
    if history:
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
        history_df['test_log_probability'] = history.test_log_probability
        history_df.to_csv(f'{MODEL_FOLDER}/history.csv', index=False)


def load_model(TARGET_PATH_TO_RESULTS, model_name):
    MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{model_name}/"
    with open(f"{MODEL_FOLDER}/model.json") as f:
        json_object = json.load(f)
        model = HiddenMarkovModel.from_json(json_object)
    return model


## Paths finding
def get_all_paths_with_probs(model, edge_value="log_probability", cutoff=None, precision=4, min_path_length=9, max_path_length=9):
    # df = pd.DataFrame(columns=["path", "prob"])
    # remove empty nodes (for speed up and only possible paths)

    new_graph, name_to_level = convert_graph_to_good_format(model,"", False, precision=precision, edge_value=edge_value)
    # +start +end
    min_path_length = min_path_length + 2
    max_path_length = max_path_length + 2
    cutoff = cutoff + 2
    print("Path finding (slow operation)...")
    path_generator = nx.all_simple_paths(new_graph, model.start.name, model.end.name, cutoff=cutoff)
    print(f"\tCondition: Took only paths with transitions above {10**-precision}...")
    print(f"\tCondition: Cutoff for path finding was {cutoff - 2}...")
    print(f"\tCondition: Took only paths of acceptable lengths: min {min_path_length - 2} , max {max_path_length - 2}...")

    paths = [path for path in tqdm(path_generator) if max_path_length >= len(path) >= min_path_length] # Might be very slow
    print(f"Paths with correct length: {len(paths)}")
    # print(f"Total paths found: {len(paths)}")
    result_paths = list()
    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    edge_to_prob = dict()
    i=0
    print(f"Calculating probs for selected paths...")
    for path in tqdm(paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        node_from = path_edges.pop(0)
        i += 1
        while path_edges:

            node_to = path_edges.pop(0)
            edge = new_graph.get_edge_data(node_from, node_to)
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            edge_to_prob[(node_from, node_to)] = edge['label']
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + edge['label']

            else:
                prob_for_path = prob_for_path * edge['label']
            # print(f"edge {edge}")
            node_from = node_to

        # print(f"Total prob for path is {prob_for_path}")

        result_paths.append(">".join(path))
        result_lens.append(len(path) - 2)
        result_probs.append(prob_for_path)
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))
        # row = {"path": path, "prob": prob_for_path}
        # df = df.append(row, ignore_index=True)
    df = pd.DataFrame({"path": result_paths, "prob": result_probs,  "normalized_prob": result_normalized_probs, "len": result_lens})
    df = df.sort_values('normalized_prob', ascending=False)
    df = df.reset_index(drop=True)

    # print("Take top 50 most probable paths")
    # df = df.head(50)
    return df, edge_to_prob


def get_state_for_name(model, state_name):
    state = [state for state in model.states if state.name == state_name][0]
    return state


def get_state_to_name_dict(model):
    d = {state.name: state for state in model.states}
    return d


# def get_prob_for_path_and_sequence(model,edge_to_prob, state_path, sequence, edge_value="log_probability"):
#     prob_for_path = 0 if edge_value == "log_probability" else 1
#     path_nodes = state_path.copy()
#     node_from = path_nodes.pop(0)
#     emissions = list(sequence)
#     while path_nodes:
#         node_to = path_nodes.pop(0)
#         tr_prob = edge_to_prob[(node_from.name, node_to.name)]
#         if node_from.name != model.start.name:
#             emission = emissions.pop(0)
#             e_prob = node_from.distribution.parameters[0][emission]
#             e_prob = np.log(e_prob) if edge_value == "log_probability" else e_prob
#             if edge_value == "log_probability":
#                 prob_for_path = prob_for_path + e_prob + tr_prob
#             else:
#                 prob_for_path = prob_for_path * e_prob * tr_prob
#
#         else:
#             if edge_value == "log_probability":
#                 prob_for_path =  prob_for_path + tr_prob
#             else:
#                 prob_for_path =  prob_for_path * tr_prob
#         # print(f"edge {edge}")
#         node_from = node_to
#     return prob_for_path

def get_emission_prob_for_path_and_sequence(path, sequence,name_to_state, start_state_name, edge_value="log_probability"):
    prob_for_path = 0 if edge_value == "log_probability" else 1
    path_nodes = path.copy()
    node_from = path_nodes.pop(0)
    emissions = list(sequence)
    while path_nodes:
        node_to = path_nodes.pop(0)
        # tr_prob = edge_to_prob[(node_from, node_to)]
        if node_from != start_state_name:
            emission = emissions.pop(0)
            e_prob = name_to_state[node_from].distribution.parameters[0][emission]
            e_prob = np.log(e_prob) if edge_value == "log_probability" else e_prob
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + e_prob
            else:
                prob_for_path = prob_for_path * e_prob

        else:
            if edge_value == "log_probability":
                prob_for_path = prob_for_path
            else:
                prob_for_path = prob_for_path
        # print(f"edge {edge}")
        node_from = node_to
    return prob_for_path

def get_peptide_to_paths_df(model, peptides, edge_value="log_probability", top_paths=None, precision=4):
    print(f"Model {model.name}")
    name_to_state = get_state_to_name_dict(model)
    print(f"\tFind all paths")
    cutoff = max(map(len, peptides))
    min_length = min(map(len, peptides))
    max_length = max(map(len, peptides))
    paths_df, edge_to_prob = get_all_paths_with_probs(model, edge_value=edge_value, cutoff=cutoff, precision=precision, min_path_length=min_length, max_path_length=max_length)

    # paths_df = paths_df.loc[paths_df['len'].between(min_length, max_length)].copy()
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths] # assumed to be sorted
        print(f"Took only {top_paths} top paths")

    # df = pd.DataFrame(columns=["peptide", "path", "prob"])
    path_to_index = {str(item.path): item.Index for item in paths_df.itertuples()}
    result_repr_matrix = np.zeros(shape=(len(peptides), len(path_to_index)), dtype=np.float)

    # make emission matrix
    name_to_index = {name: i for i, name in enumerate(name_to_state.keys())}
    acid_to_index = {acid: i for i, acid in enumerate(sorted(model.keymap[0].keys()))}
    emission_matrix = np.zeros(shape=(len(name_to_index), len(acid_to_index)))

    for name, state in name_to_state.items():
        if name != model.start.name and name != model.end.name:
            for acid, prob in state.distribution.parameters[0].items():
                emission_matrix[name_to_index[name], acid_to_index[acid]] = np.log(prob)

    print(f"\tCalculate probs for peptides")
    for _, path, tr_prob, tr_normalized_prob, path_length in tqdm(paths_df.itertuples(),total=paths_df.shape[0]):
        curr_path = path[1:]
        curr_path_indexes = [name_to_index[name] for name in curr_path]
        path_index = path_to_index[str(path)]
        for i, peptide in enumerate(peptides):
            if edge_value == "log_probability":
                emission_prob = sum([emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in zip(curr_path_indexes, peptide)])
                prob_for_path = tr_prob + emission_prob
            else:
                emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state, model.start.name, edge_value=edge_value)
                prob_for_path = tr_prob * emission_prob
            result_repr_matrix[i, path_index] = prob_for_path
            # result_paths.append(path)
            # result_probs.append(prob_for_path)

    df = pd.DataFrame(result_repr_matrix, index=peptides)
    if edge_value == "log_probability":
        df.replace([-np.inf], -55, inplace=True)
    return df, paths_df, emission_matrix, name_to_index


def get_peptide_to_paths_df_external_paths(model, peptides, paths_df, edge_value="log_probability", top_paths=None):
    print(f"Model {model.name}")
    name_to_state = get_state_to_name_dict(model)

    # paths_df = paths_df.loc[paths_df['len'].between(min_length, max_length)].copy()
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths] # assumed to be sorted
        print(f"Took only {top_paths} top paths")

    # df = pd.DataFrame(columns=["peptide", "path", "prob"])
    path_to_index = {str(item.path): item.Index for item in paths_df.itertuples()}
    result_repr_matrix = np.zeros(shape=(len(peptides), len(path_to_index)), dtype=np.float)

    # make emission matrix
    name_to_index = {name: i for i, name in enumerate(name_to_state.keys())}
    acid_to_index = {acid: i for i, acid in enumerate(sorted(model.keymap[0].keys()))}
    emission_matrix = np.zeros(shape=(len(name_to_index), len(acid_to_index)))

    for name, state in name_to_state.items():
        if name != model.start.name and name != model.end.name:
            for acid, prob in state.distribution.parameters[0].items():
                emission_matrix[name_to_index[name], acid_to_index[acid]] = np.log(prob)

    print(f"\tCalculate probs for peptides")
    for _, path, tr_prob, tr_normalized_prob, path_length in tqdm(paths_df.itertuples(),total=paths_df.shape[0]):
        curr_path = path.split(">")
        curr_path = curr_path[1:]
        curr_path_indexes = [name_to_index[name] for name in curr_path]
        path_index = path_to_index[str(path)]
        for i, peptide in enumerate(peptides):
            if edge_value == "log_probability":
                emission_prob = sum([emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in zip(curr_path_indexes, peptide)])
                prob_for_path = tr_prob + emission_prob
            else:
                emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state, model.start.name, edge_value=edge_value)
                prob_for_path = tr_prob * emission_prob
            result_repr_matrix[i, path_index] = prob_for_path / len(path)
            # result_paths.append(path)
            # result_probs.append(prob_for_path)

    df = pd.DataFrame(result_repr_matrix, index=peptides)
    if edge_value == "log_probability":
        df.replace([-np.inf], -55, inplace=True)
    return df, paths_df, emission_matrix, name_to_index


# def get_peptide_to_paths_df(model, peptides, edge_value="log_probability", top_paths=None):
#     print(f"Model {model.name}")
#     name_to_state = get_state_to_name_dict(model)
#     print(f"\tFind all paths")
#     paths_df, edge_to_prob = get_all_paths_with_probs(model, edge_value=edge_value)
#
#     if top_paths:
#         paths_df = paths_df[:top_paths] # assumed to be sorted
#         print(f"Took only {top_paths} top paths")
#     # df = pd.DataFrame(columns=["peptide", "path", "prob"])
#     result_peptides = []
#     result_paths = []
#     result_probs = []
#     print(f"\tCalculate probs for peptides")
#     for peptide in peptides:
#         for path in paths_df['path']:
#             state_path = [name_to_state[state_name] for state_name in path]
#             prob_for_path = get_prob_for_path_and_sequence(model, edge_to_prob, state_path, peptide, edge_value=edge_value)
#             result_peptides.append(peptide)
#             result_paths.append(path)
#             result_probs.append(prob_for_path)
#     df = pd.DataFrame({"peptide": result_peptides, "path":result_paths, "prob":result_probs})
#     if edge_value == "log_probability":
#         df.replace([-np.inf], -55, inplace=True)
#     return df

def get_all_viterbi_paths_for_model_and_sequences(model, peptides, edge_value="log_probability", precision=4):
    print(f"Model {model.name}")
    model_graph = model.graph

    # new_graph, name_to_level = convert_graph_to_good_format(model,"", False, precision=precision, edge_value=edge_value)
    # transition_matrix = model.dense_transition_matrix()
    print("Get viterbi paths for each peptide...")
    repeated_paths = list()
    repeated_path_state_indexes = list()

    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    result_paths = list()

    path_code_to_path = dict()

    for i, peptide in tqdm(enumerate(peptides), total=len(peptides)):
        prob_for_path, path = model.viterbi(peptide)
        path_state_indexes = "|".join([str(path_tuple[0]) for path_tuple in path]) # path as string
        path_code_to_path[path_state_indexes] = [path_tuple[1] for path_tuple in path]
    print("Select unique paths...")
    print(f"\tUnique paths: {len(path_code_to_path)}")
    unique_paths = list(path_code_to_path.values())

    for path in tqdm(unique_paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        node_from = path_edges.pop(0)
        while path_edges:
            node_to = path_edges.pop(0)
            data = model_graph.get_edge_data(node_from, node_to,default=None)
            log_probability = data['probability']
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + log_probability
            else:
                prob_for_path = prob_for_path * np.exp(log_probability)
            # print(f"edge {edge}")
            node_from = node_to
        result_paths.append(">".join([node.name for node in path]))
        result_lens.append(len(path) - 2)
        result_probs.append(prob_for_path)
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))

    paths_df = pd.DataFrame({"path": result_paths, "prob": result_probs,  "normalized_prob": result_normalized_probs, "len": result_lens})
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    return paths_df


def get_model_representation2(model, peptides, edge_value="log_probability", top_paths=None):
    assert isinstance(peptides, set)
    assert edge_value in ['log_probability', 'probability']
    peptide_to_path_df = get_peptide_to_paths_df(model, peptides, edge_value=edge_value, top_paths=top_paths)
    print("Got result dataset")
    unique_paths = peptide_to_path_df['path'].sort_values().apply(str).unique()
    path_to_index = {path : i for  i , path in enumerate(unique_paths)}
    peptide_to_path_df['path_index'] = peptide_to_path_df["path"].map(str).map(path_to_index)
    #pivot
    pivot_df = peptide_to_path_df.pivot(index='peptide', columns='path_index', values='prob')
    print("Finished")
    return pivot_df, path_to_index


def get_model_representation(model, peptides, paths_df=None, edge_value="log_probability", top_paths=None, precision=4):
    assert isinstance(peptides, set)
    assert edge_value in ['log_probability', 'probability']
    pivot_df, paths_df, emission_matrix, name_to_index = \
        get_peptide_to_paths_df_external_paths(model, peptides,  paths_df=paths_df, edge_value=edge_value, top_paths=top_paths)
    # old version with finding paths
    #     print("Precision will not be counted since we have external paths")
    #     pivot_df, paths_df, emission_matrix, name_to_index = get_peptide_to_paths_df(model, peptides,, edge_value=edge_value, top_paths=top_paths)

    # print("Got result dataset")
    # unique_paths = peptide_to_path_df['path'].sort_values().apply(str).unique()
    # path_to_index = {path : i for  i , path in enumerate(unique_paths)}
    # peptide_to_path_df['path_index'] = peptide_to_path_df["path"].map(str).map(path_to_index)
    # #pivot
    # pivot_df = peptide_to_path_df.pivot(index='peptide', columns='path_index', values='prob')
    print("Finished")
    return pivot_df, paths_df, emission_matrix, name_to_index
