import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
from sklearn.model_selection import KFold
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import CCA
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def get_units_area(areas, session):

    units_area = np.zeros(len(areas))

    for i, area in enumerate(areas):
        units_area[i] = session.units[session.units["ecephys_structure_acronym"] == area].shape[0]

    return units_area

def delete_bad_areas(session):

    areas = session.structure_acronyms
    print(len(areas))
    print(areas)

    # delete nan in areas
    while np.nan in areas:
        areas.remove(np.nan)

    # delete area with no units
    for area in areas:
        units = session.units[session.units["ecephys_structure_acronym"] == area]
        if units.shape[0] == 0:
            areas.remove(area)

    print(len(areas))
    print(areas)

    return areas

def plot_units_areas(units_areas, areas, session_ids):

    fig, axes = plt.subplots(nrows=len(session_ids), ncols=1, figsize=(12, 24))

    for i in range(len(session_ids)):
        
        axes[i].bar(range(len(areas)), units_areas[i, :], width=0.6)
        for x, y in zip(range(len(areas)), units_areas[i, :]):
            axes[i].text(x, y, y, ha='center', va='bottom', fontsize=15)
        axes[i].set_xticks(range(len(areas)))
        axes[i].set_xticklabels(areas)
        axes[i].set_ylabel(f'units', fontsize=15)
        axes[i].set_ylim(0, np.max(units_areas[i, :])+20)
        axes[i].set_title(f'session {session_ids[i]}', fontsize=15)
        axes[i].tick_params(axis='both', labelsize=15)

    fig.suptitle('units across areas in different sessions', verticalalignment='bottom', fontsize=18)

    fig.tight_layout()
    plt.show()

    return


def spike_matrix(area, stim_table, bin=0.1, period=2):
    """spike_matrix, get spike_counts using function "presentationwise_spike_counts"

    Arguments:
        area -- brain area, which need to be analyzed
        stim_table -- stimulus_table got by allensdk, which need to be analyzed

    Keyword Arguments:
        bin -- count spikes within time bin, s (default: {0.1})
        period -- the whole time period of one stimuli in drift_grating_stimuli, s (default: {2})

    Returns:
        response_matrix, shape (stims, bins, units)
    """

    area_units = session.units[session.units["ecephys_structure_acronym"] == area]

    time_bins = np.arange(0, period + bin, bin)  

    spike_counts = session.presentationwise_spike_counts(
        stimulus_presentation_ids=stim_table.index.values,  
        bin_edges=time_bins,
        unit_ids=area_units.index.values
    )
    
    response_matrix = spike_counts.values

    return response_matrix

def get_design_matrix(area, stim_table):
    """get_design_matrix design_matrix for further analysis

    Reshape response_matrix from (stims, bins, units) to (stims*bins, units)/(n_samples, units)

    Arguments:
        area -- brain area, which need to be analyzed
        stim_table -- stimulus_table got by allensdk, which need to be analyzed

    Returns:
        design_matrix, shape (n_samples, units)
    """

    response_matrix = spike_matrix(area=area, stim_table=stim_table)
    design_matrix = response_matrix.reshape(response_matrix.shape[0]*response_matrix.shape[1],
                                    response_matrix.shape[2])

    return design_matrix

def cross_val_FA(des_mat, latent_dim):

    k_fold = 10
    kf = KFold(n_splits=k_fold)
    log_like = np.zeros(k_fold)

    fold = 0
    for train_index, test_index in kf.split(des_mat):
        train, test = des_mat[train_index], des_mat[test_index]
        # print("TRAIN:", train.shape, "TEST:", test.shape)

        fa = FactorAnalysis(n_components=latent_dim)
        fa.fit(train)


        log_like[fold] = fa.score(test)   # get Average log-likelihood of test
        fold = fold + 1

    return log_like.mean()

def intra_dim_FA(des_mat):
    
    N, n_features = des_mat.shape

    # # select proper latent dimensions from [1, p-1]
    # cv_log_like = np.zeros(n_features-1)
    # for i in range(n_features-1):
    #     latent_dim = i + 1
    #     cv_log_like[i] = cross_val_FA(des_mat, latent_dim)

    # latent_dim_log_like = np.argmax(cv_log_like) + 1

    # select proper latent dimensions from [1, p-1]
    if n_features > 20:
        # as n_features is more, cv_log_like is more
        # so to save time, limit 20 possible dim to try
        start_feature = n_features - 20
        cv_log_like = np.zeros(20)
        features = range(n_features)[start_feature:]
        for i, fea in enumerate(features):
            latent_dim = fea
            cv_log_like[i] = cross_val_FA(des_mat, latent_dim)     
    else:
        cv_log_like = np.zeros(n_features-1)
        features = np.arange(n_features-1) + 1
        for i, fea in enumerate(features):
            latent_dim = fea
            cv_log_like[i] = cross_val_FA(des_mat, latent_dim)

    latent_dim_log_like = features[np.argmax(cv_log_like) + 1]

    # select proper latent dimensions with best dim got by log_like
    fa = FactorAnalysis(n_components=latent_dim_log_like)
    fa.fit(des_mat)
    load_mat = fa.components_   # (n_components, n_features)

    eigvals, eigvecs = np.linalg.eig(load_mat.T @ load_mat)
    ind = np.argsort(-eigvals)
    eigvals = eigvals[ind]
    shared_var = np.cumsum(eigvals)/np.sum(eigvals)

    threshold = 0.9
    latent_dim = np.where(shared_var>threshold)[0][0]

    return latent_dim


def get_intra_dim_area(areas, stim_table):

    intra_dim_area = np.zeros(len(areas))

    for i, area in enumerate(areas):
        des_mat = get_design_matrix(area, stim_table)
        intra_dim_area[i] = intra_dim_FA(des_mat)

    print('intra_dim_area', intra_dim_area)

    return intra_dim_area

def plot_intra_dim_areas(areas, units_area, intra_dim_area):

    # intra_dimension for all areas
    x = np.array(range(len(areas)))
    width=0.4
    f, ax = plt.subplots(figsize=(24, 12))
    ax.bar(x, units_area, width=width, label='units',fc='y')
    ax.bar(x+np.array([width]), intra_dim_area, width=width, label='intra_dimensions', tick_label=areas, fc='r')
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', labelsize=15)

    # show num on the bar
    for a, b in zip(x, units_area):
        ax.text(a, b, b, ha='center', va='bottom', fontsize=15)
    for a, b in zip(x+width, intra_dim_area):
        ax.text(a, b, b, ha='center', va='bottom', fontsize=15)

    ax.set_title(f'intra_dim & units in different areas', fontsize=15)
    plt.show()

    return

# basepath = "E:\Allensdk_data\local\ecephys_cache_dir"
basepath = "/home/jialab/Allensdk_data/local/ecephys_cache_dir/"
manifest_path = os.path.join(basepath, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
session_ids = [719161530, 750332458, 750749662, 754312389, 755434585, 756029989, 791319847, 797828357]
selected_sessions = {}

for i, session_id in enumerate(session_ids):
    session = cache.get_session_data(session_id)
    selected_sessions[session_id] = session

session = selected_sessions[755434585]
drift_stim_table = session.get_stimulus_table('drifting_gratings')
# drift_stim_table.head()

# areas = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
areas = ['VISp', 'VISrl']
# units_area = get_units_area(areas, session)
intra_dim_area = get_intra_dim_area(areas, stim_table=drift_stim_table)
# np.save('intra_dim_area_755434585', intra_dim_area)

# intra_dim_area = np.load('intra_dim_area_755434585.npy')
# plot_intra_dim_areas(areas, units_area, intra_dim_area)