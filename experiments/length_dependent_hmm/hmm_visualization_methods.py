import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pomegranate
from logomaker.src import colors
import logomaker
from logomaker.src.Glyph import Glyph
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
from scipy.stats import norm


def save_figure_to_svg(fig, dir, filename):
    fig.savefig(dir + filename, format="svg", bbox_inches='tight', transparent=False)
def save_figure_to_png(fig, dir, filename):
    fig.savefig(dir + filename, format="png", bbox_inches='tight', transparent=False)

def probs_to_info_matrix(distribution):
    amino_acids = list(distribution.parameters[0].keys())
    dummy_df = pd.DataFrame(columns = sorted(amino_acids), index=[0],  dtype=float)
    for i, (acid, prob) in enumerate(distribution.parameters[0].items()):
        dummy_df.loc[0, acid] = prob
    dummy_df.index.name = "pos"
    info_matrix = logomaker.transform_matrix(dummy_df, from_type='probability', to_type='information')
    return info_matrix



def plot_distributions_for_states(model, TARGET_PATH_TO_RESULTS, horizontal=True, discrete=True, initial_params=None):

    PATH_TO_SAVE_PICTURES = TARGET_PATH_TO_RESULTS + "/" + model.name + "/"
    if not os.path.exists(PATH_TO_SAVE_PICTURES):
        os.makedirs(PATH_TO_SAVE_PICTURES)
    for state in model.states:
        if not (state == model.start or state == model.end):
            if discrete:
                amino_acids = list(model.keymap[0].keys())
                color_dict = colors.get_color_dict(None, amino_acids)
                if horizontal:
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 3))

                    glyph_list = []
                    for i, (acid, prob) in enumerate(state.distribution.parameters[0].items()):
                        #print(i, acid, prob)
                        floor = 0
                        ceiling = floor + abs(prob)
                        this_color = color_dict[acid]
                        flip = False
                        glyph = Glyph(i, acid,
                                      ax=ax,
                                      floor=floor,
                                      width=1,
                                      ceiling=ceiling,
                                      color=this_color,
                                      flip=flip,
                                      zorder=0,
                                      # font_name='Arial Rounded MT Bold',
                                      alpha=1.0,
                                      vpad=0.0)
                        glyph_list.append(glyph)
                        xmin = min([g.p - .5*g.width for g in glyph_list])
                        xmax = max([g.p + .5*g.width for g in glyph_list])
                        ax.set_xlim([xmin, xmax])
                        # set ylims
                        ymin = min([g.floor for g in glyph_list])
                        # ymax = max([g.ceiling for g in glyph_list])
                        ymax = 1
                        ax.set_ylim([ymin, ymax])
                        plt.xticks(np.arange(0, 20, 1.0), labels=sorted(amino_acids))

                        save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}.svg")
                        plt.close(fig)
                else:
                    dummy_df = pd.DataFrame(columns = sorted(amino_acids), index = [0],  dtype=float)
                    for i, (acid, prob) in enumerate(state.distribution.parameters[0].items()):
                        dummy_df.loc[0, acid] = prob
                    dummy_df.index.name = "pos"
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 9))
                    # return dummy_df
                    info_matrix = logomaker.transform_matrix(dummy_df, from_type='probability', to_type='information')
                    ww_logo = logomaker.Logo(info_matrix,ax=ax,
                                             # font_name='Arial Rounded MT Bold',
                                             #stack_order='small_on_top'
                                             )
                    save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}.svg")
                    plt.close(fig)
            elif type(state.distribution) is pomegranate.IndependentComponentsDistribution:
                list_of_dists = state.distribution.parameters[0]
                fig, axes = plt.subplots(ncols=1, nrows=len(list_of_dists), figsize=(5, len(list_of_dists)* 3))
                for i, distr in enumerate(list_of_dists):

                    distr_params = distr.parameters
                    mean_value = distr_params[0]
                    std_value = distr_params[1]

                    mean_for_picture = mean_value
                    std_for_picture = std_value
                    if initial_params is not None:
                        assert type(initial_params) == list
                        initial_distr = initial_params[i]
                        initial_mean = initial_distr['mean']
                        initial_std = initial_distr['std']
                        mean_for_picture = initial_mean
                        std_for_picture = max(initial_std, std_for_picture) + abs(mean_value-initial_mean)
                    x = np.linspace(mean_for_picture-4*std_for_picture, mean_for_picture+4*std_for_picture, 1000)
                    y1= norm.pdf(x, loc=mean_value, scale=std_value)
                    axes[i].plot(x, y1, 'b', lw=2, label=f'pdf')
                    if initial_params is not None:
                        y2= norm.pdf(x, loc=initial_mean, scale=initial_std)
                        axes[i].plot(x, y2, 'r', lw=2, label=f'pdf_initial')
                    axes[i].text(0.0, 0.75, f'{np.round(mean_value,3)}', fontsize=35, transform=axes[i].transAxes)
                    axes[i].text(0.7, 0.75, f'{np.round(std_value,3)}', fontsize=35, transform=axes[i].transAxes)
                save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}.svg")
                plt.close(fig)
            else:
                # normal distr here
                distr_params = state.distribution.parameters
                mean_value = distr_params[0]
                std_value = distr_params[1]
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 3))
                x = np.linspace(mean_value-4*std_value, mean_value+4*std_value, 1000)
                y = norm.pdf(x, loc=mean_value, scale=std_value)
                ax.plot(x, y, 'r', lw=2, label='pdf')
                save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}.svg")
                plt.close(fig)



def plot_motifs_for_paths(model, TARGET_PATH_TO_RESULTS, paths_df, top=None):
    PATH_TO_SAVE_PICTURES = TARGET_PATH_TO_RESULTS + "/" + model.name + "/paths/"
    if not os.path.exists(PATH_TO_SAVE_PICTURES):
        os.makedirs(PATH_TO_SAVE_PICTURES)
    result_df = pd.DataFrame()
    state_name_to_index = dict()
    for i,state in enumerate(model.states):
        state_name_to_index[state.name] = i
    states = model.states
    if top:
        paths_df = paths_df.sort_values("prob", ascending=False)[:top]
    for index, row in paths_df.iterrows():
        probs_url = f"path_{index}_probs.png"
        info_url = f"path_{index}_info.png"
        result_dict = {
            "id": index,
            "info_picture": '<img src="'+ info_url + '" width="320" >',
            "probs_picture": '<img src="'+ probs_url + '" width="320" >',
            "motif_url": info_url,
            "prob": row['prob']
        }

        df = pd.DataFrame()
        for position, state_name in enumerate(row['path'], start=-1):
            if state_name not in [model.start.name, model.end.name]:
                params = states[state_name_to_index[state_name]].distribution.parameters[0]
                df = df.append(params, ignore_index=True)
                result_dict[f'pos_{position}'] = state_name

        info_matrix = logomaker.transform_matrix(df, from_type='probability', to_type='information')
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
        info_logo = logomaker.Logo(info_matrix, ax=ax)
        save_figure_to_png(fig, dir=PATH_TO_SAVE_PICTURES, filename=info_url)
        plt.close(fig)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
        probs_logo = logomaker.Logo(df, ax=ax)
        save_figure_to_png(fig, dir=PATH_TO_SAVE_PICTURES, filename=probs_url)
        plt.close(fig)


        result_df = result_df.append(result_dict, ignore_index=True)
    result_df.to_html(f'{PATH_TO_SAVE_PICTURES}/paths.html', escape=False)

    return result_df

def make_levels(model, precision=4):

    source_gr = model.graph
    name_to_level = dict()
    name_to_size = dict()
    passed_nodes = set()
    nodes_stack = []
    nodes_stack.append(model.start)
    name_to_level[model.start.name] = 1

    while len(nodes_stack) > 0:
        cur_node = nodes_stack.pop()
        passed_nodes.add(cur_node)
        for node_to in source_gr.successors(cur_node):
            data = source_gr.get_edge_data(cur_node, node_to, default=None)
            log_probability = data['probability']
            p = np.exp(log_probability)
            p = round(p, precision)
            if node_to not in passed_nodes:
                if p != 0:
                    cur_level = name_to_level[cur_node.name] + 1
                    if not node_to.name in name_to_level:
                        name_to_level[node_to.name] = cur_level
                        #print(cur_node.name, "->", node_to.name, p,  cur_level )
                    nodes_stack.append(node_to)

    # repeat for all subgraphs
    not_seen_states = []
    for state in model.states:
        if state not in passed_nodes:
            not_seen_states.append(state)
    state_to_in_count = dict()
    for state in not_seen_states:
        predecessors = source_gr.predecessors(state)
        selected_predecessors = []
        for node_from in predecessors:
            data = source_gr.get_edge_data(node_from, state, default=None)
            log_probability = data['probability']
            p = np.exp(log_probability)
            p = round(p, precision)
            if p != 0:
                selected_predecessors.append(node_from)
        state_to_in_count[state.name] = len(selected_predecessors)
    min_in_count_states = [state for state in not_seen_states if state_to_in_count[state.name] == min(state_to_in_count.values())]
    for state in min_in_count_states:
        if state not in passed_nodes:
            print("new starting point")
            nodes_stack.append(state)
            name_to_level[state.name] = 1
            while len(nodes_stack) > 0:
                cur_node = nodes_stack.pop()
                passed_nodes.add(cur_node)
                for node_to in source_gr.successors(cur_node):
                    data = source_gr.get_edge_data(cur_node, node_to, default=None)
                    log_probability = data['probability']
                    p = np.exp(log_probability)
                    p = round(p, precision)
                    if node_to not in passed_nodes:
                        if p != 0:
                            cur_level = name_to_level[cur_node.name] + 1
                            if not node_to.name in name_to_level:
                                name_to_level[node_to.name] = cur_level
                                #print(cur_node.name, "->", node_to.name, p,  cur_level )
                            nodes_stack.append(node_to)

    return name_to_level

def convert_graph_to_good_format(model, logo_pictures_path, use_logos=True, precision = 4, edge_value="probability"):
    print(f"Converting graph for model ({model.name}) (Delete zero edges)")
    assert edge_value in ["probability", "log_probability"]
    name_to_level = make_levels(model, precision)    # print(name_to_level)
    source_gr = model.graph
    gr = nx.DiGraph()

    # calculate information for sizes
    per_state_information = dict()
    per_level_information = defaultdict(float)
    for state in model.states:
        if state not in [model.start, model.end]:
            if type(state.distribution) == pomegranate.DiscreteDistribution:
                info_matrix = probs_to_info_matrix(state.distribution)
                per_state_information[state.name] = sum(info_matrix.loc[0])
                per_level_information[name_to_level[state.name]] += per_state_information[state.name]
            elif type(state.distribution) is pomegranate.NormalDistribution:
                per_state_information[state.name] = state.distribution.parameters[1]
                per_level_information[name_to_level[state.name]] += per_state_information[state.name]
            elif type(state.distribution) is pomegranate.IndependentComponentsDistribution:
                per_state_information[state.name] = 1
                per_level_information[name_to_level[state.name]] += per_state_information[state.name]

    # Add all states from model's graph (logo pictures should be prepared)
    for state in model.states:
        if state == model.start or state == model.end or not use_logos: # start and end states without picture
            gr.add_node(state.name,
                        level=name_to_level[state.name],
                        size=35)
        else:
            # info_portion = per_state_information[state.name] / per_level_information[name_to_level[state.name]]
            size_percentage = per_state_information[state.name] / max(per_state_information.values())
            gr.add_node(state.name,
                        level=name_to_level[state.name],
                        size=size_percentage * 35,
                        shape='image',
                        #image=f"file:///{logo_pictures_path}{state.name}.svg")
                        image=f"{logo_pictures_path}{state.name}.svg")

    # Add all non-zero transitions
    for edge in source_gr.edges:
        data = source_gr.get_edge_data(edge[0], edge[1], default=None)
        log_probability = data['probability']
        p = np.exp(log_probability)
        p = round(p, precision)
        if p != 0:
            if edge_value == "probability":
                gr.add_edge(edge[0].name, edge[1].name, label=p)
            else:
                gr.add_edge(edge[0].name, edge[1].name, label=log_probability)

    return gr, name_to_level


def make_pyviz_graph(model, TARGET_PATH_TO_RESULTS, precision=4):
    graph, name_to_level = convert_graph_to_good_format(model, "", precision=precision)
                                                      # f"{os.getcwd()}/{TARGET_PATH_TO_RESULTS}/{model.name}/")

    nt = Network('500px', '1600', directed=True)
    nt.from_nx(graph)
    nt.show_buttons()
    #nt.show_buttons(filter_=['physics'])
    nt.save_graph(f"{TARGET_PATH_TO_RESULTS}\\{model.name}\\pyviz-pr-{precision}.html")
    # nt.show(f"{TARGET_PATH_TO_RESULTS}\\{model.name}\\pyviz.html")
    return nt


def make_logos_for_train_sequences(target_length, per_allele_per_kfold_per_length_training_data, TARGET_PATH_TO_RESULTS):
    per_allele_per_split_prob_matrix = dict()
    for allele_name in per_allele_per_kfold_per_length_training_data.keys():
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
        per_allele_per_split_prob_matrix[allele_name] = dict()
        for split_num, per_length_data in per_allele_per_kfold_per_length_training_data[allele_name].items():
            model_name = f"{target_length}-mer_model-{target_allele_name}-{split_num}"
            binders_array = per_length_data[target_length]
            matrix = logomaker.alignment_to_matrix(binders_array)
            info_matrix = logomaker.transform_matrix(matrix, from_type='counts', to_type='information')
            prob_matrix = logomaker.transform_matrix(matrix, from_type='counts', to_type='probability')
            ww_logo = logomaker.Logo(info_matrix,
                                     # font_name='Arial Rounded MT Bold',
                                     )
            ww_logo.ax.set_title(f"Allele {allele_name}, length {target_length}, split {split_num}")
            PATH_TO_SAVE_PICTURES = TARGET_PATH_TO_RESULTS + f"{model_name}/"
            save_figure_to_svg(fig=ww_logo.fig,  dir=PATH_TO_SAVE_PICTURES, filename=f"logo_for_sequences-{target_length}.svg")



"""

def statistics_for_letter(target_length, letter, positions):
    per_allele_per_split_selected_peptides = dict()
    for allele_name in ALLELES:
        per_allele_per_split_selected_peptides[allele_name] = dict()
        for split_num, per_length_data in per_allele_per_kfold_per_length_binders_train[allele_name].items():
            binders_array = per_length_data[target_length]
            selected_peptides = np.array([peptide for peptide in binders_array  for position in positions if peptide[position] == letter])
            per_allele_per_split_selected_peptides[allele_name][split_num] = selected_peptides
    return per_allele_per_split_selected_peptides

statistics_L_0 = statistics_for_letter(8, 'L', [0])
statistics_L_0_unique = {allele:{split: set(arr) for split, arr in statistics_L_0[allele].items()} for allele in ALLELES}
statistics_L_1 = statistics_for_letter(8, 'L', [1])
statistics_L_1_unique = {allele:{split: set(arr) for split, arr in statistics_L_1[allele].items()} for allele in ALLELES}

print("L on a first position:", [len(arr) for arr in statistics_L_0[ALLELES[0]].values()], "Unique", [len(set(arr)) for arr in statistics_L_0_unique[ALLELES[0]].values()])
print('\t', statistics_L_0[ALLELES[0]][0][:6])

print("L on a second position:", [len(arr) for arr in statistics_L_1[ALLELES[0]].values()], "Unique", [len(set(arr)) for arr in statistics_L_1_unique[ALLELES[0]].values()])
print('\t', statistics_L_1[ALLELES[0]][0][:6])
print("L on both positions:", [len(np.intersect1d(arr1, arr2)) for arr1, arr2 in zip(statistics_L_0[ALLELES[0]].values(), statistics_L_1[ALLELES[0]].values())])
print('\t',np.intersect1d(statistics_L_1[ALLELES[0]][0],statistics_L_0[ALLELES[0]][0])[:6])
"""
