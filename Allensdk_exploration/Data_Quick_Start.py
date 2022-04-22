import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

data_directory = '/local/ecephys_cache_dir' # must be updated to a valid directory in your filesystem
# data_directory = "D:\local\ecephys_cache_dir"

manifest_path = os.path.join(data_directory, "manifest.json")

print(manifest_path)

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path, timeout = 20000000 * 60)

sessions = cache.get_session_table()

# downloaded_sessions = [715093703, 719161530, 721123822, 732592105, 737581020, 739448407, 742951821, 743475441, 744228101, 746083955,
#                        750332458, 750749662, 751348571, 754312389, 754829445, 755434585, 756029989, 757216464, 757970808, 758798717,
#                        759883607, 760345702, 760693773, 761418226, 762120172, 762602078, 763673393, 766640955, 767871931, 768515987,
#                        771160300, 771990200, 773418906, 774875821, 778240327, 778998620, 779839471, 781842082, 786091066, 787025148,
#                        789848216, 791319847, 793224716, 794812542, 797828357, 798911424, 799864342, 816200189, 819186360, 819701982,
#                        821695405, 829720705, 831882777, 835479236, 839068429, 839557629, 840012044, 847657808]

for session_id, row in sessions.iterrows():

    # if session_id in downloaded_sessions:
    #     print("downloaded_session:", session_id)
    #     continue

    truncated_file = True
    directory = os.path.join(data_directory + '/session_' + str(session_id))
    
    while truncated_file:

        print("Start session:", session_id)
        session = cache.get_session_data(session_id)
        print("End session:", session_id)

        try:
            print(session.specimen_name)
            truncated_file = False
        except OSError:
            shutil.rmtree(directory)
            print(" Truncated spikes file, re-downloading")

    for probe_id, probe in session.probes.iterrows():
        
        print(' ' + probe.description)
        truncated_lfp = True
        
        while truncated_lfp:
            try:

                print("Start probe:", probe_id)
                session = cache.get_session_data(session_id)
                print("End probe:", probe_id)

                lfp = session.get_lfp(probe_id)
                truncated_lfp = False
            except OSError:
                fname = directory + '/probe_' + str(probe_id) + '_lfp.nwb'
                os.remove(fname)
                print("  Truncated LFP file, re-downloading")
            except ValueError:
                print("  LFP file not found.")
                truncated_lfp = False