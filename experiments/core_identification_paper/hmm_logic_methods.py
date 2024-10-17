import types
from pomegranate import *
import json
import numpy as np
import pandas as pd
from pomegranate.io import BatchedDataGenerator, SequenceGenerator
from hmm_visualization_methods import convert_graph_to_good_format
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

MINUS_INF_CONSTANT = -20
MINUS_INF_CONSTANT_NORMALIZED = -0.5


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
                stop_threshold=5e-5,
                use_test=True,
                step_to_save_model=30
                ):
    per_split_model = dict()
    per_split_history = dict()

    assert type(prepared_models) in [dict, type(
        None)], "Prepared models should be either None (for new models) or per split models dict"

    for split_num, per_length_data in per_kfold_per_length_data.items():
        per_length_current_weights = per_kfold_per_length_weights_for_data[split_num]
        per_length_test_data = per_kfold_per_length_test_data[split_num]
        per_length_counts = {target_length: len(per_length_data[target_length]) for target_length in target_lengths}

        # list_of_arrays = [per_length_data[target_length] for target_length in target_lengths]
        # list_of_arrays_for_weights = [np.array(per_length_current_weights[target_length]) for target_length in
        #                               target_lengths]
        # list_of_test_arrays = [per_length_test_data[target_length] for target_length in target_lengths]
        # binders_array = np.concatenate(list_of_arrays)
        # weights_array = np.concatenate(list_of_arrays_for_weights)
        #
        # binders_test_array = np.concatenate(list_of_test_arrays)

        binders_array = np.array([per_length_data[target_length][i] for target_length in target_lengths
                                  for i in range(len(per_length_data[target_length]))], dtype=object)
        weights_array = np.array([per_length_current_weights[target_length][i] for target_length in target_lengths
                                  for i in range(len(per_length_current_weights[target_length]))], dtype=object)

        binders_test_array = np.array([per_length_test_data[target_length][i] for target_length in target_lengths
                                       for i in range(len(per_length_test_data[target_length]))], dtype=object)
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
        sample_X_test = np.array([[char for char in item] for item in binders_test_array], dtype=object)
        if minibatch_training:
            sequence_generator = BatchedDataGenerator(sample_X, batches_per_epoch=batches_per_epoch,
                                                      batch_size=batch_size)
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
            sequence_test_generator = SequenceGenerator(sample_X_test) if use_test else None
            print(f"Simple training: Train Length {len(sequence_generator)}, "
                  f"init_batch_size {init_batch_size}")
            if init_batch_size is not None:
                assert init_batch_size < len(
                    sequence_generator), f"init batch size should be {len(sequence_generator)} or lower " \
 \
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
                # "batches": batches_per_epoch,
                # "batch_size": batch_size,
                # "init_batch_size": init_batch_size,
                # "lr": lr_decay,
                # "edge_inertia": edge_inertia,
                # "distribution_inertia": distribution_inertia,
                # "emission_pseudocount": emission_pseudocount,
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
                                                            use_pseudocount=use_pseudocount,
                                                            # inertia
                                                            n_jobs=n_jobs,
                                                            algorithm=algorithm,
                                                            weights=weights_array
                                                            # labels
                                                            # weights  An array of weights, one for each sequence to train on. If None, all sequences are equally weighted
                                                            )
            model_name = f"{name_postfix}-{params_string}_model-{target_allele_name}-{split_num}"
            model.name = model_name
        else:
            print("Example of data", sample_X[0])

            print("Use pseudocount is", use_pseudocount)
            model = prepared_models[split_num]
            model, history = model.fit(sample_X,
                                       sequences_test=sequence_test_generator,
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
                                       min_iterations=min_iterations,
                                       max_iterations=max_iterations,
                                       emission_pseudocount=emission_pseudocount,
                                       transition_pseudocount=transition_pseudocount,
                                       use_pseudocount=use_pseudocount,
                                       weights=weights_array
                                       )
        per_split_model[split_num] = model
        per_split_model[split_num] = model
        per_split_history[split_num] = history
        break
    return per_split_model, per_split_history




# Add states to existing model
def add_more_states_and_reset_transitions(source_model, ADD_STATES_NUM, new_sates_names=[], new_name="new_model",
                                          new_probs="fair"):
    old_start_index = source_model.start_index
    old_end_index = source_model.end_index
    old_states = source_model.states.copy()
    old_state_name_to_index = {state.name: i for i, state in enumerate(old_states)}
    old_states_names = {state.name for state in old_states}
    old_transitions = source_model.dense_transition_matrix()
    state_distributions = []
    model = source_model.copy()

    print(f"Initial length {len(model.states)}")
    keymap = model.keymap[0]
    for i in range(ADD_STATES_NUM):
        d = DiscreteDistribution({key: prob for key, prob in zip(keymap, np.random.uniform(0, 1, size=len(keymap)))})
        state_distributions.append(d)

    state_names = new_sates_names
    states_to_add = [State(distribution, name=name) for name, distribution in zip(state_names, state_distributions)]
    print(f"num states to add {len(states_to_add)}")
    model.add_states(states_to_add)
    model.name = new_name
    model.start.name = f"{new_name}-start"
    model.end.name = f"{new_name}-end"
    model.bake(verbose=True, merge=None)  # merge None because we don't want to delete the states without nodes
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
                prob = median_from_start_prob  # we want to have fair prob(median)
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
                from_probs.append(
                    old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]])
        if from_state.name in old_states_names:
            median_from_prob = np.mean(from_probs)

        for j, to_state in enumerate(states):
            if from_state.name in old_states_names and to_state.name in old_states_names:
                prob = old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
            elif from_state.name in old_states_names:
                if new_probs == "fair":
                    prob = median_from_prob  # probabilities from old to new will be median of all from probabilities
                    # prob = 1 / ADD_STATES_NUM  # probabilities from old to new will be 1/(num_new)
                elif new_probs == "tiny":
                    prob = 0.04
                else:
                    raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
            else:  # from new state
                if to_state.name in old_states_names:
                    prob = 1.0  # to old state prob is 1
                else:
                    prob = 0  # from new to old prob is 0
            if prob != 0:
                # print(f"ADD transition from {from_state.name} {to_state.name} with prob {prob}")
                model.add_transition(from_state, to_state, prob)

    # Connect states to the end of the model if a non-zero probability
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[old_state_name_to_index[state.name], old_end_index]
        else:
            prob = 1.0  # if from new state  equal probs from all the states + end state
        if prob != 0:
            # print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            model.add_transition(state, model.end, prob)
    model.bake(verbose=False)
    return model



def init_distribution(amino_acids=None, initial_means=None, initial_stds=None, discrete_distribution=False, equal_probs=False, anchor=False, anchor_max_components=5,   anchor_default_pseudocount=50,):
    amino_acids_list = list(sorted(set(amino_acids)))  # IMPORTANT. SHould be set because HMM method will convert to set and there willbe different order (non alphabetic)
    if not discrete_distribution:
        components_d  = list()
        for i in range(len(initial_means)):
            mean_value = initial_means[i]
            std_value = initial_stds[i]
            d = NormalDistribution(mean_value, std_value)
            components_d.append(d)
        total_d = IndependentComponentsDistribution(components_d)
    else:
        if not equal_probs:
            d_dict = {key: prob for key, prob in zip(amino_acids_list, np.random.default_rng().uniform(0, 1, size=len(amino_acids_list)))}
        else:
            d_dict = {key: 1 for key in amino_acids_list}

        d_sum = sum(d_dict.values())
        d_dict = {key: value/d_sum for key, value in d_dict.items()}
        if anchor:
            total_d = DiscreteDistributionAnchor(d_dict)
            total_d.max_components = anchor_max_components
            total_d.default_pseudocount = anchor_default_pseudocount
        else:
            total_d = DiscreteDistribution(d_dict)
    return total_d


def init_grouped_structure_model_joiners_and_cycles_here(
        states_per_group=5, # may be int or list of numbers for each group
        number_of_groups=5,
        amino_acids="ACDEFGHIKLMNPQRSTVWY",
        freeze_distr=False,
        CYCLE_PSEUDOCOUNT=1,
        initial_params=None,
        start_cycle=True,
        end_cycle=True,
        self_cycles=False,
        discrete_distribution=False,
        equal_probs_for_cycle=False,
        tie_cycle_states=False,
        anchor_max_components=5,
        anchor_default_pseudocount=50,
        anchor_states=True,
        model_complexity=1,
        tie_anchor_states=False,
        intermediate_cycles=False,
        equal_probs_for_intermediate_cycle=False,
        tie_intermediate_cycles=False,
        freeze_int_cycle_distr=False
):
    INTRAGROUP_WEIGHT = 1
    INTERGROUP_WEIGHT = 1

    ##  0.16  and 0.83 situation both ends
    START_WEIGHT = 1
    TO_FIRST_CYCLE_WEIGHT = 5
    TO_LAST_CYCLE_WEIGHT = 5


    if start_cycle or end_cycle:
        TO_FIRST_JOINER_WEIGHT = 5#just to make transitions even inside groups
        TO_LAST_JOINER_WEIGHT = 1
    else:
        TO_FIRST_JOINER_WEIGHT = 1
        TO_LAST_JOINER_WEIGHT = 2.5

    # IN_FIRST_CYCLE_WEIGHTS = np.linspace(0.1, 0.9, num=model_complexity)
    IN_FIRST_CYCLE_WEIGHTS = np.repeat(0.5, model_complexity)
    FROM_FIRST_CYCLE_WEIGHTS =  1 - IN_FIRST_CYCLE_WEIGHTS
    # IN_LAST_CYCLE_WEIGHTS = np.linspace(0.1, 0.9, num=model_complexity)[::-1]
    IN_LAST_CYCLE_WEIGHTS = np.repeat(0.5, model_complexity)
    FROM_LAST_CYCLE_WEIGHTS = 1 - IN_LAST_CYCLE_WEIGHTS
    END_WEIGHT = 1
    

    
    ## 0.5 and 0.5 situation both ends
#     START_WEIGHT = 5
#     TO_FIRST_CYCLE_WEIGHT = 5
#     TO_LAST_CYCLE_WEIGHT = 5


#     if start_cycle or end_cycle:
#         TO_FIRST_JOINER_WEIGHT = 5#just to make transitions even inside groups
#         TO_LAST_JOINER_WEIGHT = 1
#     else:
#         TO_FIRST_JOINER_WEIGHT = 1
#         TO_LAST_JOINER_WEIGHT = 2.5

#     IN_FIRST_CYCLE_WEIGHT = TO_FIRST_JOINER_WEIGHT
#     IN_LAST_CYCLE_WEIGHT = 1
#     END_WEIGHT = 5
#     FROM_LAST_CYCLE_WEIGHT = 1
    
    
    
    amino_acids_list = list(amino_acids)


    joiner_positions = {0: [i for i in range(model_complexity)], 1: [i for i in range(model_complexity, model_complexity*2)]}
    number_of_joiners = sum(len(positions) for positions in joiner_positions.values())
    number_of_groups = number_of_groups

    if type(states_per_group) == list:
        assert number_of_groups == len(states_per_group), f"Number of groups is expected to be {number_of_groups} but got {len(states_per_group)} in a list of states per group"
    else:
        states_per_group = [states_per_group for i in range(number_of_groups)]
    distributions = list()


    mean_values = list()
    stds = list()
    if initial_params:
        for initial_distr_params in initial_params:
            mean_values.append(initial_distr_params['mean'])
            stds.append(initial_distr_params['std'])
        print(f"mean values : {mean_values}")
        print(f"std values : {stds}")

    # initialize group states
    for group_number in range(number_of_groups):
        states_in_group = states_per_group[group_number]
        for state_num in range(states_in_group):
            total_d = init_distribution(amino_acids=amino_acids,
                                        initial_means=mean_values,
                                        initial_stds=stds,
                                        discrete_distribution=discrete_distribution)
            distributions.append(total_d)
    #initialize joiner states (random here)
    joiner_label_to_index_in_model = dict()
    
    for joiner_name in range(int(number_of_joiners/2)):
        # d = DiscreteDistribution(
        #     {key: prob for key, prob in zip(amino_acids_list, np.random.uniform(0, 1, size=len(amino_acids_list)))})
        total_d = init_distribution(amino_acids=amino_acids,
                                    initial_means=mean_values,
                                    initial_stds=stds,
                                    discrete_distribution=discrete_distribution, 
                                    anchor=anchor_states, 
                                    anchor_max_components=anchor_max_components,
                                    anchor_default_pseudocount=anchor_default_pseudocount)
        joiner_label_to_index_in_model[joiner_name] = len(distributions)
        distributions.append(total_d)
        
    # second half (tie if necessary
    for joiner_name in range(int(number_of_joiners/2), number_of_joiners):
        # d = DiscreteDistribution(
        #     {key: prob for key, prob in zip(amino_acids_list, np.random.uniform(0, 1, size=len(amino_acids_list)))})
        if tie_anchor_states:
            pair_joiner_name = number_of_joiners - joiner_name - 1
         
            #print(f"original name {joiner_name}, pair {pair_joiner_name}")
            total_d = distributions[joiner_label_to_index_in_model[pair_joiner_name]]
        else:
            total_d = init_distribution(amino_acids=amino_acids,
                                        initial_means=mean_values,
                                        initial_stds=stds,
                                        discrete_distribution=discrete_distribution, 
                                        anchor=anchor_states, 
                                        anchor_max_components=anchor_max_components,
                                        anchor_default_pseudocount=anchor_default_pseudocount)
        joiner_label_to_index_in_model[joiner_name] = len(distributions)
        distributions.append(total_d)
    
    # initialize cycle states
    number_of_cycles = 0
    cycle_label_to_index_in_model = dict()

    if start_cycle:
        # for each joiner its own cycle
        # Same distribution for all cycles
        start_cycle_d = init_distribution(amino_acids=amino_acids,
                                        initial_means=mean_values,
                                        initial_stds=stds,
                                        discrete_distribution=discrete_distribution, equal_probs=equal_probs_for_cycle)
        
        
        
        for start_cycle_name in (joiner_positions[0]):
            if tie_cycle_states:
                start_d = start_cycle_d
            else:
                start_d = init_distribution(amino_acids=amino_acids,
                                            initial_means=mean_values,
                                            initial_stds=stds,
                                            discrete_distribution=discrete_distribution, equal_probs=equal_probs_for_cycle)

            if freeze_distr:
                start_d.freeze()

            cycle_label_to_index_in_model[start_cycle_name] = len(distributions)
            distributions.append(start_d)
            number_of_cycles += 1
    if end_cycle:
       # Same distribution for all cycles
        start_cycle_d = init_distribution(amino_acids=amino_acids,
                                  initial_means=mean_values,
                                    initial_stds=stds,
                                    discrete_distribution=discrete_distribution, equal_probs=equal_probs_for_cycle)
        
        for end_cycle_name in joiner_positions[max(joiner_positions.keys())]:
            if tie_cycle_states:
                end_d = start_cycle_d
            else:
                end_d = init_distribution(amino_acids=amino_acids,
                                      initial_means=mean_values,
                                        initial_stds=stds,
                                        discrete_distribution=discrete_distribution, equal_probs=equal_probs_for_cycle)
            if freeze_distr:
                end_d.freeze()

            cycle_label_to_index_in_model[end_cycle_name] = len(distributions)
            distributions.append(end_d)
            number_of_cycles += 1
    number_of_int_cycles = 0
    if intermediate_cycles:
        number_of_int_cycles = 1
        # initialize cycle states
        # here only one cycle
        int_cycle_name = 0
        int_cycle_label_to_index_in_model = dict()
        # Same distribution for all cycles
        int_cycle_d = init_distribution(amino_acids=amino_acids,
                                  initial_means=mean_values,
                                    initial_stds=stds,
                                    discrete_distribution=discrete_distribution, equal_probs=equal_probs_for_intermediate_cycle)
        int_cycle_label_to_index_in_model[int_cycle_name] = len(distributions)
        if freeze_int_cycle_distr:
            int_cycle_d.freeze()
        distributions.append(int_cycle_d)
        



    # print("joiners positions", joiner_label_to_index_in_model)

    num_states = sum(states_per_group) + number_of_joiners +  number_of_cycles + number_of_int_cycles# groups, joiners, 2 cycles
    print(f"states will be groups:{sum(states_per_group)} + joiners:{number_of_joiners} + start/end cycles:{ number_of_cycles} + int_cycles:{number_of_int_cycles}")

    assert num_states == len(distributions)

    state_names = ["s{:03d}".format(i) for i in range(num_states)]
    transition_matrix = numpy.zeros((num_states, num_states))

    # start probabilities
    first_joiners = set(joiner_label_to_index_in_model[label] for label in joiner_positions[0])
    last_joiners = set(joiner_label_to_index_in_model[label] for label in joiner_positions[max(joiner_positions.keys())])
    # connect start cycle to the first joiners

    start_cycles = set(cycle_label_to_index_in_model[label] for label in joiner_positions[0] if start_cycle) # because each joiner has it's own cycle
    end_cycles = set(cycle_label_to_index_in_model[label] for label in joiner_positions[max(joiner_positions.keys())] if end_cycle) # because each joiner has it's own cycle
    


    # start only for first cycle and first joiner
    start_probabilities = [START_WEIGHT
                           if i in start_cycles or i in first_joiners
                           else 0
                           for i in range(num_states)]


    # do not restrict ending in multiple states, so probs are equal to transitions inside

    end_probabilities = [END_WEIGHT
                         if i in end_cycles or i in last_joiners
                         else 0
                         for i in range(num_states)]

    #  start cycle transitions
    if start_cycle:
        for start_cycle_idx, joiner_idx in zip(start_cycles, first_joiners):
            transition_matrix[start_cycle_idx][joiner_idx] = TO_FIRST_JOINER_WEIGHT # from cycle to joiners
            # for joiner_idx in first_joiners:
            #     transition_matrix[start_cycle_idx][joiner_idx] = TO_FIRST_JOINER_WEIGHT # from cycle to joiners

    # connect last joiners to last cycle
    if end_cycle:
        for end_cycle_idx, joiner_idx in zip(end_cycles, last_joiners):
            transition_matrix[joiner_idx][end_cycle_idx] = TO_LAST_CYCLE_WEIGHT
            # for joiner_idx in last_joiners:
            #     transition_matrix[joiner_idx][end_cycle_idx] = TO_LAST_CYCLE_WEIGHT
    if intermediate_cycles:
        for int_cycle_index in int_cycle_label_to_index_in_model.values():
             transition_matrix[int_cycle_index][int_cycle_index] = 1

    # Transitions inside groups
    for group_number in range(number_of_groups):
        states_in_group = states_per_group[group_number]
        for i in range(states_in_group):
            state_num_from = sum(states_per_group[:group_number]) + i
            # fill transitions inside each group with INTRAGROUP_WEIGHT
            if intermediate_cycles:         
                transition_matrix[state_num_from][int_cycle_label_to_index_in_model[0]] = 1
                transition_matrix[int_cycle_label_to_index_in_model[0]][state_num_from] = 1
            for j in range(states_in_group):

                state_num_to =  sum(states_per_group[:group_number]) + j
                if state_num_from != state_num_to:
                    transition_matrix[state_num_from][state_num_to] = INTRAGROUP_WEIGHT

            if self_cycles:
                transition_matrix[state_num_from][state_num_from] = INTRAGROUP_WEIGHT

            # transition to next group
            if group_number != number_of_groups - 1:
                for j in range(states_in_group):
                    state_num_to = sum(states_per_group[:group_number+1]) + j
                    transition_matrix[state_num_from][state_num_to] = INTERGROUP_WEIGHT
                    #through the cycle




    # connect first joiner to first group
    for joiner_index in first_joiners:
        group_number = 0
        state_num_from = joiner_index
        states_in_group =  states_per_group[group_number]
        for j in range(states_in_group):
            state_num_to = sum(states_per_group[:group_number]) + j
            transition_matrix[state_num_from][state_num_to] = INTERGROUP_WEIGHT
        #and to intermediate cycle 
        if intermediate_cycles:
            transition_matrix[state_num_from][int_cycle_label_to_index_in_model[0]] = INTERGROUP_WEIGHT

    # connect last group to the last joiner
    for joiner_index in last_joiners:
        group_number = number_of_groups -1
        state_num_to = joiner_index
        states_in_group =  states_per_group[group_number]
        for j in range(states_in_group):
            state_num_from = sum(states_per_group[:group_number]) + j
            transition_matrix[state_num_from][state_num_to] = TO_LAST_JOINER_WEIGHT
        #and to intermediate cycle 
        if intermediate_cycles:
            transition_matrix[int_cycle_label_to_index_in_model[0]][state_num_to] = TO_LAST_JOINER_WEIGHT
    model_name = 'grouped_model'


    # assert sum(start_probabilities) == len(joiner_positions[0])* START_WEIGHT + TO_CYCLE_WEIGHT, start_probabilities
    # assert sum(end_probabilities) ==  (len(joiner_positions[max(joiner_positions.keys())]) + 1)  * END_WEIGHT, end_probabilities + list(last_joiners)

    model = HiddenMarkovModel.from_matrix(
        name=model_name,
        transition_probabilities=transition_matrix,
        starts=start_probabilities,
        distributions=distributions,
        ends=end_probabilities,
        state_names=state_names,
        verbose=False)
    # Update created model with specific pseudocounts

    if start_cycle:
        CURR_CYCLE_INDEX = 0
        for start_cycle_idx, joiner_idx in zip(start_cycles, first_joiners):
            model.add_transition(model.states[start_cycle_idx], model.states[start_cycle_idx], IN_FIRST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=CYCLE_PSEUDOCOUNT)
            model.add_transition(model.start, model.states[start_cycle_idx], TO_FIRST_CYCLE_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)
            model.add_transition(model.states[start_cycle_idx], model.states[joiner_idx], FROM_FIRST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=CYCLE_PSEUDOCOUNT)
            CURR_CYCLE_INDEX+=1
        #
        # for joiner_idx in first_joiners:
        #         model.add_transition(model.states[start_cycle_idx], model.states[joiner_idx], TO_FIRST_JOINER_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)
    if end_cycle:
        CURR_CYCLE_INDEX = 0
        for end_cycle_idx, joiner_idx in zip(end_cycles, last_joiners):
            model.add_transition(model.states[end_cycle_idx], model.states[end_cycle_idx], IN_LAST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=CYCLE_PSEUDOCOUNT)
            model.add_transition(model.states[end_cycle_idx],model.end, FROM_LAST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=CYCLE_PSEUDOCOUNT)
            model.add_transition(model.states[joiner_idx], model.states[end_cycle_idx], TO_LAST_CYCLE_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)
            CURR_CYCLE_INDEX+= 1
            # for joiner_idx in last_joiners:
            #     model.add_transition(model.states[joiner_idx], model.states[end_cycle_idx], TO_LAST_CYCLE_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)
        # model.add_transition(model.states[end_cycle_idx], model.states[end_cycle_idx], IN_CYCLE_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)

    model.bake(verbose=False)
    return model







# Add states to existing model
def add_more_states_to_the_end(source_model, ADD_STATES_NUM, new_sates_names=[], new_name="new_model", new_probs="fair",
                               edge_pseudocount=None):
    old_start_index = source_model.start_index
    old_end_index = source_model.end_index
    old_states = source_model.states.copy()
    old_state_name_to_index = {state.name: i for i, state in enumerate(old_states)}
    old_states_names = {state.name for state in old_states}
    old_transitions = source_model.dense_transition_matrix()
    state_distributions = []
    model = source_model.copy()

    print(f"Initial length {len(model.states)}")
    keymap = model.keymap[0]
    for i in range(ADD_STATES_NUM):
        d = DiscreteDistribution({key: prob for key, prob in zip(keymap, np.random.uniform(0, 1, size=len(keymap)))})
        state_distributions.append(d)

    state_names = new_sates_names
    states_to_add = [State(distribution, name=name) for name, distribution in zip(state_names, state_distributions)]
    print(f"num states to add {len(states_to_add)}")
    model.add_states(states_to_add)
    model.name = new_name
    model.start.name = f"{new_name}-start"
    model.end.name = f"{new_name}-end"
    model.bake(verbose=True, merge=None)  # merge None because we don't want to delete the states without nodes
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
                    prob = old_transitions[
                        old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
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
            else:  # from new state
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
            # print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            model.add_transition(state, model.end, prob, pseudocount=edge_pseudocount)
    model.bake(verbose=False)
    return model



def save_model(TARGET_PATH_TO_RESULTS, model, history=None):
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
def get_all_paths_with_probs(model, edge_value="log_probability", cutoff=None, precision=4, min_path_length=9,
                             max_path_length=9):
    # df = pd.DataFrame(columns=["path", "prob"])
    # remove empty nodes (for speed up and only possible paths)

    new_graph, name_to_level = convert_graph_to_good_format(model, "", False, precision=precision,
                                                            edge_value=edge_value)
    # +start +end
    min_path_length = min_path_length + 2
    max_path_length = max_path_length + 2
    cutoff = cutoff + 2
    print("Path finding (slow operation)...")
    path_generator = nx.all_simple_paths(new_graph, model.start.name, model.end.name, cutoff=cutoff)
    print(f"\tCondition: Took only paths with transitions above {10 ** -precision}...")
    print(f"\tCondition: Cutoff for path finding was {cutoff - 2}...")
    print(
        f"\tCondition: Took only paths of acceptable lengths: min {min_path_length - 2} , max {max_path_length - 2}...")

    paths = [path for path in tqdm(path_generator) if
             max_path_length >= len(path) >= min_path_length]  # Might be very slow
    print(f"Paths with correct length: {len(paths)}")
    # print(f"Total paths found: {len(paths)}")
    result_paths = list()
    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    edge_to_prob = dict()
    i = 0
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
    df = pd.DataFrame(
        {"path": result_paths, "prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens})
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

def get_emission_prob_for_path_and_sequence(path, sequence, name_to_state, start_state_name,
                                            edge_value="log_probability"):
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
    paths_df, edge_to_prob = get_all_paths_with_probs(model, edge_value=edge_value, cutoff=cutoff, precision=precision,
                                                      min_path_length=min_length, max_path_length=max_length)

    # paths_df = paths_df.loc[paths_df['len'].between(min_length, max_length)].copy()
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths]  # assumed to be sorted
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
    for _, path, tr_prob, tr_normalized_prob, path_length in tqdm(paths_df.itertuples(), total=paths_df.shape[0]):
        curr_path = path[1:]
        curr_path_indexes = [name_to_index[name] for name in curr_path]
        path_index = path_to_index[str(path)]
        for i, peptide in enumerate(peptides):
            if edge_value == "log_probability":
                emission_prob = sum([emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in
                                     zip(curr_path_indexes, peptide)])
                prob_for_path = tr_prob + emission_prob
            else:
                emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state, model.start.name,
                                                                        edge_value=edge_value)
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
    paths_df = paths_df.sort_values('norm_viterbi_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths]  # assumed to be sorted
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

    for _, path,  tr_prob, path_length, mean_viterbi, mean_bw_probs, path_pep_cnt  in tqdm(paths_df.itertuples(),
                                                                                              total=paths_df.shape[0]):
        curr_path = path.split(">")
        curr_path = curr_path[1:]
        curr_path_indexes = [name_to_index[name] for name in curr_path]
        path_index = path_to_index[str(path)]
        for i, peptide in enumerate(peptides):
            if edge_value == "log_probability":

                if len(peptide) == len(curr_path_indexes) - 1:  # -1 because we have end state at the end
                    emission_prob = sum(
                        [emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in
                         zip(curr_path_indexes, peptide)])
                    prob_for_path = tr_prob + emission_prob
                else:
                    prob_for_path = -np.nan
            else:
                if len(peptide) == len(curr_path_indexes):
                    emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state,
                                                                            model.start.name, edge_value=edge_value)
                    prob_for_path = tr_prob * emission_prob

                else:
                    prob_for_path = 0

            result_repr_matrix[i, path_index] = prob_for_path
            # result_paths.append(path)
            # result_probs.append(prob_for_path)

    df = pd.DataFrame(result_repr_matrix, index=peptides)
    # if edge_value == "log_probability":
    #     df.replace([-np.inf], MINUS_INF_CONSTANT_NORMALIZED, inplace=True)
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
#             prob_for_path = get_prob_for_path_and_sequence(convert_graph_to_good_formatmodel, edge_to_prob, state_path, peptide, edge_value=edge_value)
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
    path_code_to_cnt = defaultdict(int)
    path_code_to_mean_viterbi_prob = defaultdict(float)

    for i, peptide in tqdm(enumerate(peptides), total=len(peptides)):
        prob_for_path, path = model.viterbi(peptide)
        path_state_indexes = "|".join([str(path_tuple[0]) for path_tuple in path])  # path as string
        path_code_to_path[path_state_indexes] = [path_tuple[1] for path_tuple in path]
        n = path_code_to_cnt[path_state_indexes]
        mean_viterbi = path_code_to_mean_viterbi_prob[path_state_indexes]
        mean_viterbi = (mean_viterbi * n + prob_for_path) / (n + 1)
        path_code_to_mean_viterbi_prob[path_state_indexes] = mean_viterbi
        path_code_to_cnt[path_state_indexes] += 1

    print("Select unique paths...")
    print(f"\tUnique paths: {len(path_code_to_path)}")
    unique_paths = list(path_code_to_path.values())
    mean_viterbies = list(path_code_to_mean_viterbi_prob.values())
    cnts = list(path_code_to_cnt.values())
    for path in tqdm(unique_paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        node_from = path_edges.pop(0)
        while path_edges:
            node_to = path_edges.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
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

    paths_df = pd.DataFrame(
        {"path": result_paths, "prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens,
         "mean_viterbi": mean_viterbies, "peptide_cnts": cnts})
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    return paths_df

def generate_radom_paths_for_model_mark_core(model, n, per_model_important_states,edge_value="log_probability"):
    name_to_state = get_state_to_name_dict(model)
    name_to_index = {name: i for i, name in enumerate(name_to_state.keys())}
    acid_to_index = {acid: i for i, acid in enumerate(sorted(model.keymap[0].keys()))}
    emission_matrix = np.zeros(shape=(len(name_to_index), len(acid_to_index)))
    for name, state in name_to_state.items():
        if name != model.start.name and name != model.end.name:
            for acid, prob in state.distribution.parameters[0].items():
                emission_matrix[name_to_index[name], acid_to_index[acid]] = np.log(prob)
    print(f"Model {model.name}")
    model_graph = model.graph
    result_tr_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    result_paths = list()
    result_simplified_paths = list()
    result_viterbi_probs = list()
    result_total_probs = list()
    result_bw_probs = list()
    result_start_core = list()
    result_end_core = list()
    peptides = list()
    print(f"Generate {n} peptides and paths")
    for i in range(n):
        peptide_arr, path = model.sample(path=True)
        peptide = "".join(peptide_arr)
        peptides.append(peptide)
        viterbi_prob, viterbi_path = model.viterbi(peptide)
        bw_prob = model.log_probability(peptide)
        tr_prob = 0 if edge_value == "log_probability" else 1
        path_nodes = [node for node in path]
        path_length = len(path_nodes)
        result_paths.append(">".join([node.name for node in path_nodes]))
        count_al = [i for i,node in enumerate(path_nodes[1:]) if node.name in per_model_important_states[model.name]  ]
        start_core = count_al[0] if len(count_al) > 0 else np.nan
        end_core = count_al[-1] if len(count_al) > 0 else np.nan
        curr_path = path_nodes[1:]
        curr_path_indexes = [name_to_index[node.name] for node in curr_path]

        probs_dif = 1- np.round((viterbi_prob-bw_prob)/ path_length,2)


        if np.isnan(start_core) or np.isnan(end_core):
            simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for node in path_nodes[1:-1]]).astype(np.float64)
        else:
            simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for node in path_nodes[1:-1]]).astype(np.float64)
            assert start_core is not np.nan
            assert end_core is not np.nan
            # simplified_path[start_core:end_core+1] = probs_dif
        result_simplified_paths.append(simplified_path)
        node_from = path_nodes.pop(0)
        while path_nodes:
            node_to = path_nodes.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
            log_probability = data['probability']
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            if edge_value == "log_probability":
                tr_prob = tr_prob + log_probability
            else:
                tr_prob = tr_prob * np.exp(log_probability)
            # print(f"edge {edge}")
            node_from = node_to


        if edge_value == "log_probability":
            emission_prob = sum(
                [emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in
                 zip(curr_path_indexes, peptide)])
            prob_for_path = tr_prob + emission_prob
        else:
                emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state,
                                                                        model.start.name, edge_value=edge_value)
                prob_for_path = tr_prob * emission_prob

        result_lens.append(path_length - 2)
        result_tr_probs.append(tr_prob)
        result_normalized_probs.append(tr_prob / path_length)
        result_viterbi_probs.append(viterbi_prob)
        result_bw_probs.append(bw_prob)
        result_total_probs.append(prob_for_path)
        result_start_core.append(start_core)
        result_end_core.append(end_core)
    paths_df = pd.DataFrame(
        {"path": result_paths,
         "simplified_path": result_simplified_paths,
         "core_start": result_start_core, "core_end": result_end_core,
         "tr_prob": result_tr_probs, "normalized_prob": result_normalized_probs, "len": result_lens,
         "viterbi_probs": result_viterbi_probs,
         "bw_probs": result_bw_probs,
         "total_probs": result_total_probs}, index=peptides)
    return paths_df


def get_all_viterbi_paths_for_model_and_sequences_mark_core(model, peptides, per_model_important_states,
                                                            per_model_skip_states=None,
                                                            edge_value="log_probability", precision=4,
                                                            unique_paths=True, generate_peptides=False,
                                                            generate_count=50000, fill_whole_core=False, cumulative_metric=False,
                                                            ):
    """
    Wil leave only last peptide in the peptide df if path was duplicated by multiple peptides
    :param model:
    :param peptides:
    :param per_model_important_states:
    :param edge_value:
    :param precision:
    :return:
    """
    print(f"Model {model.name}")
    model_graph = model.graph
    if per_model_important_states == None:
        mark_core = False
        print("Warning: no important states are not defined, simplified path column will be absent")
    else:
        mark_core = True
    if mark_core:
        assert unique_paths is False, "If you want core estimates then you want each individual peptide to be counted, not unique paths"

    if generate_peptides:
        assert peptides is None, "If you want to generate peptides -- you don't need peptides as argument"
        peptides = list()

    print("Get viterbi paths for each peptide...")

    result_probs = list()
    result_lens = list()
    result_paths = list()
    result_simplified_paths = list()
    result_peptides = list()
    result_viterbi_probs = list()
    result_bw_probs = list()
    result_start_core = list()
    result_end_core = list()
    result_cnts = list()

    path_code_to_path = dict()
    path_code_to_cnt = defaultdict(int)
    path_code_to_total_viterbi_prob = defaultdict(float)
    path_code_to_total_bw_prob = defaultdict(float)
    path_code_to_peptide = defaultdict(str)

    peptide_count = generate_count if generate_peptides else len(peptides)

    for i in tqdm(range(peptide_count)):
        if not generate_peptides:  # peptides are given, calculate all from peptide
            peptide = peptides[i]
            viterbi_prob, path = model.viterbi(peptide)
            path = [path_tuple[1] for path_tuple in path]
            bw_prob = model.log_probability(peptide)
        else:
            peptide_arr, path = model.sample(path=True)
            peptide = "".join(peptide_arr)
            peptides.append(peptide)
            viterbi_prob, viterbi_path = model.viterbi(peptide)
            bw_prob = model.log_probability(peptide)

        if unique_paths:
            path_index = "|".join([state.name for state in path])  # path as string
        else:
            path_index = peptide
            path_code_to_peptide[path_index] = peptide
        path_code_to_path[path_index] = path
        path_code_to_total_viterbi_prob[path_index] += viterbi_prob
        path_code_to_total_bw_prob[path_index] += bw_prob
        path_code_to_cnt[path_index] += 1
    print("Subselect unique paths (if unique path argument passed)..")
    print(f"\tUnique paths: {len(path_code_to_path)} ")
    # unique_paths = list(path_code_to_path.values())
    # mean_viterbies = list(viterbi_prob/ cnt for viterbi_prob, cnt in zip(path_code_to_total_viterbi_prob.values(), path_code_to_cnt.values()))
    # mean_bws = list(bw_prob/ cnt for bw_prob, cnt in zip(path_code_to_total_bw_prob.values(), path_code_to_cnt.values()))

    for path_code, path_nodes in tqdm(path_code_to_path.items()):
        if mark_core:
            count_al = [i for i,node in enumerate(path_nodes[1:]) if node.name in per_model_important_states[model.name]]
            start_core = count_al[0] if len(count_al) > 0 else np.nan
            end_core = count_al[-1] if len(count_al) > 0 else np.nan
        peptide_cnt = path_code_to_cnt[path_code]
        viterbi_prob = path_code_to_total_viterbi_prob[path_code] / peptide_cnt
        bw_prob = path_code_to_total_bw_prob[path_code] / peptide_cnt
        path_length = len(path_nodes)



        tr_prob_for_path = 0 if edge_value == "log_probability" else 1
        result_paths.append(">".join([node.name for node in path_nodes]))
        path_nodes_copy = path_nodes.copy()

        node_from = path_nodes_copy.pop(0)
        i=0
        cumulative_probs = np.zeros(len(path_nodes) -2)
        cummulative_prob = 0

        while path_nodes_copy:
            node_to = path_nodes_copy.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
            log_probability = data['probability']
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            if edge_value == "log_probability":
                tr_prob_for_path = tr_prob_for_path + log_probability
                if mark_core:
                    peptide = path_code_to_peptide[path_code]
                    if node_to.distribution:
                        cummulative_prob = cummulative_prob + log_probability + np.log(node_to.distribution.parameters[0][peptide[i]])
                        cumulative_probs[i] = cummulative_prob
                        i += 1
            else:
                tr_prob_for_path = tr_prob_for_path * np.exp(log_probability)
            # print(f"edge {edge}")
            node_from = node_to


        # normalize by peptide length
        peptide_len = path_length - 2
        viterbi_prob = viterbi_prob / peptide_len
        bw_prob = bw_prob / peptide_len
        tr_prob_for_path = tr_prob_for_path / peptide_len


        # probs_dif = viterbi_prob

        if mark_core:
            probs_dif = 1
            #probs_dif = 1 - np.round(viterbi_prob-bw_prob,2)
            cur_path =  path_nodes[1:-1]
            if np.isnan(start_core) or np.isnan(end_core):
                simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for node in cur_path]).astype(np.float64)

            else:
                simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for node in cur_path]).astype(np.float64)
                assert start_core is not np.nan
                assert end_core is not np.nan
                if fill_whole_core:
                    
                    simplified_path[start_core:end_core] = [probs_dif if cur_path[i].name not in per_model_skip_states[model.name] else 0 for i in range(start_core,end_core)]
                    
                    if cumulative_metric:
                        cumulative_probs = cumulative_probs / peptide_len
                    # simplified_path[start_core:end_core] = cumulative_probs[start_core:end_core]
                        simplified_path = cumulative_probs

            result_simplified_paths.append(simplified_path)
            result_start_core.append(start_core)
            result_end_core.append(end_core)
        result_lens.append(peptide_len)
        result_probs.append(tr_prob_for_path)
        result_viterbi_probs.append(viterbi_prob)
        result_bw_probs.append(bw_prob)

        result_cnts.append(path_code_to_cnt[path_code])
        if not unique_paths:
            result_peptides.append(path_code_to_peptide[path_code])
    if unique_paths:
        paths_df = pd.DataFrame(
            {"path": result_paths,
             "norm_tr_prob": result_probs, "len": result_lens,
             "norm_viterbi_prob": result_viterbi_probs,
             "norm_bw_prob": result_bw_probs,
             "peptide_cnt": result_cnts})
    else:
        paths_df = pd.DataFrame(
        {"path": result_paths,
         "norm_tr_prob": result_probs, "len": result_lens,
         "norm_viterbi_prob": result_viterbi_probs,
         "norm_bw_prob": result_bw_probs,
         "peptide_cnt": result_cnts}, index=result_peptides)
    if mark_core:
        paths_df['simplified_path'] = result_simplified_paths
        paths_df['core_start'] = result_start_core
        paths_df['core_end'] = result_end_core

    return paths_df


def calculate_model_core_guesses(allele_peptides,models_dict,per_model_important_states, per_model_skip_states=None, selected_peptides=None):
    total_df = pd.DataFrame(index=allele_peptides)
    selected_peptide_to_paths = defaultdict(list)

    for i, (model_key, model) in enumerate(models_dict.items()):
        viterbi_paths = get_all_viterbi_paths_for_model_and_sequences_mark_core(model, allele_peptides, per_model_important_states, per_model_skip_states=per_model_skip_states, unique_paths=False, fill_whole_core=True, cumulative_metric=False)
        if 'simplified_path' not in total_df.columns:
            total_df['simplified_path'] = viterbi_paths['simplified_path']
       
        else:
            total_df['simplified_path'] = total_df['simplified_path'].values + viterbi_paths['simplified_path'].values
            
        total_df[f'core_start_{i}'] =  viterbi_paths['simplified_path'].apply(lambda x: np.where(np.array(x) > 0)[0]).apply(lambda x: x[0] if len(x) > 0 else 0)
        total_df[f'core_end_{i}'] =  viterbi_paths['simplified_path'].apply(lambda x: np.where(np.array(x) > 0)[0]).apply(lambda x: x[-1] if len(x)> 0 else 1)
        total_df[f'core_{i}'] =  viterbi_paths['simplified_path']
        total_df[f'prob_{i}'] =  viterbi_paths['norm_viterbi_prob']
        
        if selected_peptides:
            for selected_peptide in selected_peptides:
                selected_peptide_to_paths[selected_peptide].append(viterbi_paths.loc[selected_peptide]['simplified_path'])
    total_df['peptide'] = total_df.index.values

    return total_df, selected_peptide_to_paths







def get_all_viterbi_paths_for_model_and_sequences_preserve_peptides_check_importance_masks(model, peptides, per_model_important_states, edge_value="log_probability", precision=4):
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
    result_peptides = list()
    result_simplified_paths = list()

    path_code_to_path = dict()
    path_code_to_cnt = defaultdict(int)
    path_code_to_mean_viterbi_prob = defaultdict(float)


    for i, peptide in tqdm(enumerate(peptides), total=len(peptides)):
        prob_for_path, path = model.viterbi(peptide)
        path_state_indexes = "|".join([str(path_tuple[0]) for path_tuple in path])  # path as string
        path_code_to_path[path_state_indexes] = [path_tuple[1] for path_tuple in path]
        n = path_code_to_cnt[path_state_indexes]
        mean_viterbi = path_code_to_mean_viterbi_prob[path_state_indexes]
        mean_viterbi = (mean_viterbi * n + prob_for_path) / (n + 1)
        path_code_to_mean_viterbi_prob[path_state_indexes] = mean_viterbi
        path_code_to_cnt[path_state_indexes] += 1

    print("Select unique paths...")
    print(f"\tUnique paths: {len(path_code_to_path)}")
    unique_paths = list(path_code_to_path.values())
    mean_viterbies = list(path_code_to_mean_viterbi_prob.values())
    cnts = list(path_code_to_cnt.values())
    for path in tqdm(unique_paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        saved_path_nodes = path_edges.copy()
        node_from = path_edges.pop(0)
        while path_edges:
            node_to = path_edges.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
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
        result_simplified_paths.append(np.array([1 if node.name in per_model_important_states[model.name] else 0 for node in saved_path_nodes[1:]]))
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))

    paths_df = pd.DataFrame(
        {"path": result_paths, "simplified_path": result_simplified_paths ,"prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens,
         "mean_viterbi": mean_viterbies, "peptide_cnts": cnts})
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    return paths_df

def get_all_viterbi_paths_for_model_and_sequences_preserve_peptides(model, peptides, edge_value="log_probability", precision=4):
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
    result_peptides = list()

    path_code_to_path = dict()
    path_code_to_cnt = defaultdict(int)
    path_code_to_mean_viterbi_prob = defaultdict(float)

    
    for i, peptide in tqdm(enumerate(peptides), total=len(peptides)):
        prob_for_path, path = model.viterbi(peptide)
        path_state_indexes = "|".join([str(path_tuple[0]) for path_tuple in path])  # path as string

        path_code_to_path[path_state_indexes] = [path_tuple[1] for path_tuple in path]
        n = path_code_to_cnt[path_state_indexes]
        mean_viterbi = path_code_to_mean_viterbi_prob[path_state_indexes]
        mean_viterbi = (mean_viterbi * n + prob_for_path) / (n + 1)
        path_code_to_mean_viterbi_prob[path_state_indexes] = mean_viterbi
        path_code_to_cnt[path_state_indexes] += 1

    print("Select unique paths...")
    print(f"\tUnique paths: {len(path_code_to_path)}")
    unique_paths = list(path_code_to_path.values())
    mean_viterbies = list(path_code_to_mean_viterbi_prob.values())
    cnts = list(path_code_to_cnt.values())
    for path in tqdm(unique_paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        node_from = path_edges.pop(0)
        while path_edges:
            node_to = path_edges.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
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

    paths_df = pd.DataFrame(
        {"path": result_paths, "prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens,
         "mean_viterbi": mean_viterbies, "peptide_cnts": cnts})
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    return paths_df


def get_model_representation2(model, peptides, edge_value="log_probability", top_paths=None):
    assert isinstance(peptides, set)
    assert edge_value in ['log_probability', 'probability']
    peptide_to_path_df = get_peptide_to_paths_df(model, peptides, edge_value=edge_value, top_paths=top_paths)
    print("Got result dataset")
    unique_paths = peptide_to_path_df['path'].sort_values().apply(str).unique()
    path_to_index = {path: i for i, path in enumerate(unique_paths)}
    peptide_to_path_df['path_index'] = peptide_to_path_df["path"].map(str).map(path_to_index)
    # pivot
    pivot_df = peptide_to_path_df.pivot(index='peptide', columns='path_index', values='prob')
    print("Finished")
    return pivot_df, path_to_index


def get_model_representation(model, peptides, paths_df=None, edge_value="log_probability", top_paths=None, precision=4):
    assert len(peptides) == len(set(peptides))
    assert edge_value in ['log_probability', 'probability']
    pivot_df, paths_df, emission_matrix, name_to_index = \
        get_peptide_to_paths_df_external_paths(model, peptides, paths_df=paths_df, edge_value=edge_value,
                                               top_paths=top_paths)
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

