import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import FactorAnalysis
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


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

areas = ['VISp', 'VISrl']

import time
start = time.time()
intra_dim_area = get_intra_dim_area(areas, stim_table=drift_stim_table)
stop = time.time()
print(f"Time per iteration: {stop - start}s")

# no program running: 5%CPU, 0%MEM, 5.73G/504G
# no retrict cpu kernels: 4800%CPU, 7.95G/504G Mem, time 1415s
# OMP_NUM_THREADS=2: 200%CPU, 0.5%MEM, 7.98G/504G, time 1605s
# OMP_NUM_THREADS=8: 800%CPU, 0.5%MEM, 8.00G/504G, time 1368s
# OMP_NUM_THREADS=16: 1600%CPU, 0.5%MEM, 8.07G/504G, time 1356s
# OMP_NUM_THREADS=32: 3200%CPU, 0.5%MEM, 8.16G/504G, time 1403s
# OMP_NUM_THREADS=64: 4800%CPU, 0.5%MEM, 8.31G/504G, time 1616s
# OMP_NUM_THREADS=90: 4800%CPU, 0.5%MEM, 8.32G/504G, time 1597s. (Error: nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS" (64))