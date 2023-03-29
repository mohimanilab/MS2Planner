import logging
import sys
import os
import numpy as np
import pandas as pd

logger = logging.getLogger('MS2Planner.baseline')


def ReadFile(infile_name, sample_name, bg_name, suffix):
    if sample_name is not None and bg_name is not None:
        full_feat = pd.read_csv(infile_name)
        rt_mz_feature_id = {}
        sample_intensity_col = 'DATAFILE:'+sample_name+':'+suffix
        background_intensity_col = 'DATAFILE:'+bg_name+':'+suffix
        rt = np.array(full_feat['RT']).reshape(-1, 1)
        mz = np.array(full_feat['m/z']).reshape(-1, 1)
        charge = np.array(full_feat['Charge']).reshape(-1, 1)
        feature_id = np.array(full_feat['ID']).reshape(-1, 1)
        sample_intensity = np.array(
            full_feat[sample_intensity_col]).reshape(-1, 1)
        bg_intensity = np.array(
            full_feat[background_intensity_col]).reshape(-1, 1)
        for i in range(len(rt)):
            rt_mz_feature_id[(rt[i, 0], mz[i, 0])] = feature_id[i, 0]
        return np.hstack((mz, rt, charge, bg_intensity, sample_intensity)), rt_mz_feature_id
    data = np.genfromtxt(infile_name, delimiter=",", skip_header=1)
    return data, None


def DataFilter_old(data, intensity, intensity_ratio):
    data = data[data[:, 4] != 0]  # remove samples with intensity = 0
    # remove samples with intensity < given
    data = data[data[:, 4] >= intensity]
    data = data[
        data[:, 4] / (data[:, 3] + 1e-4) > intensity_ratio
    ]  # remove samples with intensity ratio < given
    print(data)
    return data


def DataFilter(data, intensity, intensity_ratio, max_same_RT):
    df = pd.DataFrame(data)  # convert NumPy array to pandas DataFrame
    df = df[df[4] != 0]  # remove samples with intensity = 0
    df = df[df[4] >= intensity]  # remove samples with intensity < given
    df = df[df[4] / (df[3] + 1e-4) > intensity_ratio]  # remove samples with intensity ratio < given
          
    # Here we are limiting the number of features with the exact same RT to max_same_RT
    initial_nb_features = df.shape[0]-1
    logger.info('   Initial number of features = '+str(initial_nb_features))
    df = df.sort_values(by=[1, 4], ascending=[True, False])
    df = df.groupby(1).head(max_same_RT)
    afterfiltering_nb_features = df.shape[0]-1
    logger.info('   Remaining features = '+str(afterfiltering_nb_features)+' after same RT filtering with top '+str(max_same_RT))
    return df.values  # convert pandas DataFrame back to NumPy array


def PathGen(data, window_len, num_path, iso, delay):
    window_len += delay
    start = min(data[:, 1])
    end = max(data[:, 1])
    total_features = data.shape[0]
    path_features = [0] * num_path
    path = []
    while start < end:
        curr_end = start + window_len
        tmp_data = data[(data[:, 1] >= start) & (data[:, 1] < curr_end)]
        if len(tmp_data) != 0:
            ind = np.argsort(tmp_data[:, 4])
            ind = ind[::-1]
            tmp = []
            for i in range(num_path):
                if i >= len(ind):
                    break
                tmp.append(
                    (
                        tmp_data[ind[i], 0],
                        iso,
                        window_len,
                        start + delay / 2,
                        curr_end - delay / 2,
                        tmp_data[ind[i], 4],
                        tmp_data[ind[i], 1],
                        tmp_data[ind[i], 2],
                    )
                )
                path_features[i] += 1
            path.append(tmp)
        start = curr_end
    logger.info(
        'Total number of features: '+str(total_features))
    logger.info(
        'Maximum number of iterative experiments: '+str(num_path))
    for i in range(num_path):
        if path_features[i] > 0:  # return only path that are are more than zero feature
            total_features -= path_features[i]
            logger.info(
                "[%d/%d max]: features: %d, rest: %d"
                % (i + 1, num_path, path_features[i], total_features)
            )

    return path

def WriteFile(outfile_name, path, num_path):
    #logger.info('WriteFile')
    #logger.info(num_path)
    for i in range(num_path):
        file_path = outfile_name[:-4]+'_path_'+str(i+1)+'.csv'
        with open(file_path, "wt", encoding='utf-8', newline='\n') as text_file:
            n = text_file.write('Mass [m/z],mz_isolation,duration,rt_start,rt_end,intensity,rt_apex,charge\n')
            for j in range(len(path)):
                if i > len(path[j]) - 1:
                    continue
                n = text_file.write(str(path[j][i]).replace('(', '').replace(')', '') + "\n")
        
        # Check if the file has only one line and delete
        with open(file_path, "rt", encoding="utf-8", newline="\n") as text_file:
            lines = text_file.readlines()
            if len(lines) == 1:
                os.remove(file_path)



def WriteFileFormatted(file_name, path, num_path, rt_mz_feature_id):
    paths, mzs, isos, starts, ends, ints, rts, charges, durs, feats = [
    ], [], [], [], [], [], [], [], [], []
    for i in range(num_path):
        for j in range(len(path)):
            if i > len(path[j]) - 1:
                continue
            paths.append(i)
            mzs.append(path[j][i][0])
            isos.append(path[j][i][1])
            durs.append(path[j][i][2])
            starts.append(path[j][i][3])
            ends.append(path[j][i][4])
            ints.append(path[j][i][5])
            rts.append(path[j][i][6])
            charges.append(path[j][i][7])
            feats.append(rt_mz_feature_id[(rts[-1], mzs[-1])])
    d = {'path': paths, 'ID': feats, 'Mass [m/z]': mzs, 'mz_isolation': isos, 'duration': durs,
         'rt_start': starts, 'rt_end': ends, 'intensity': ints, 'rt_apex': rts, 'charge': charges}
    df = pd.DataFrame(data=d)
    df.to_csv(path_or_buf=file_name, sep=',', index=True, encoding='ISO-8859-1')
