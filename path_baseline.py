import logging
import sys

import numpy as np

logger = logging.getLogger('path_finder.baseline')

def ReadFile(infile_name):
    data = np.genfromtxt(infile_name, delimiter=",", skip_header=1)
    return data


def DataFilter(data, intensity, intensity_ratio):
    data = data[data[:, 4] != 0]  # remove samples with intensity = 0
    data = data[data[:, 4] >= intensity]  # remove samples with intensity < given
    data = data[
    data[:, 4] / (data[:, 3] + 1e-4) > intensity_ratio
    ]  # remove samples with intensity ratio < given
    return data

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
    
    for i in range(num_path):
        total_features -= path_features[i]
        logger.info(
            "[%d/%d]: features: %d, rest: %d"
            % (i + 1, num_path, path_features[i], total_features)
        )
    return path


def WriteFile(outfile_name, path, num_path):
    text_file = open(outfile_name, "wt")
    for i in range(num_path):
        n = text_file.write("path" + str(i) + "\t")
        for j in range(len(path)):
            if i > len(path[j]) - 1:
                continue
            n = text_file.write(
                "{:.4f}".format(path[j][i][0])
                + " "
                + "{:.4f}".format(path[j][i][1])
                + " "
                + "{:.4f}".format(path[j][i][2])
                + " "
                + "{:.4f}".format(path[j][i][3])
                + " "
                + "{:.4f}".format(path[j][i][4])
                + " "
                + "{:.4f}".format(path[j][i][5])
                + " "
                + "{:.4f}".format(path[j][i][6])
                + " "
                + str(path[j][i][7])
                + "\t"
            )
        n = text_file.write("\n")
    text_file.close()
