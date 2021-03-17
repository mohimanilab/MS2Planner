import logging

import numpy as np

logger = logging.getLogger('path_finder.apex')

# this function reads the apex file (.csv) and stores in a numpy array
# first row is ignored
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


def NodeEdge1Create(data, intensity_accu, delay, min_time, max_time):
    num_node = 0
    rt_node_dic = {}
    node_rt_dic = {}
    edge_intensity_dic = {}
    edge = []
    for i in range(len(data)):
        num_node += 1
        rt_isolation = intensity_accu / data[i, 4]
        rt_isolation = min(max_time, max(rt_isolation, min_time))
        logger.debug(rt_isolation)
        left_node = data[i, 1] - 0.5 * rt_isolation
        right_node = data[i, 1] + 0.5 * rt_isolation + delay
        if left_node in rt_node_dic.keys():
            rt_node_dic[left_node].append(num_node)
        else:
            rt_node_dic[left_node] = [num_node]
        node_rt_dic[num_node] = (left_node, data[i, 0], data[i, 2])
        num_node += 1
        if right_node in rt_node_dic.keys():
            rt_node_dic[right_node].append(num_node)
        else:
            rt_node_dic[right_node] = [num_node]
        node_rt_dic[num_node] = (right_node, data[i, 0], data[i, 2])
        edge.append([num_node - 1, num_node, -1])
        edge_intensity_dic[(left_node, right_node)] = data[i, 4]
    return num_node, rt_node_dic, node_rt_dic, edge, edge_intensity_dic


def Edge0Create(num_node, rt_node_dic, edge):
    rt_node = []
    for i in rt_node_dic.keys():
        rt_node.append(i)
    rt_node.sort()

    for i in range(len(rt_node)):
        if i == 0:
            for j in rt_node_dic[rt_node[i]]:
                edge.append([0, j, 0])
        elif i == len(rt_node) - 1:
            for j in rt_node_dic[rt_node[i]]:
                edge.append([j, num_node + 1, 0])
        else:
            for j in rt_node_dic[rt_node[i]]:
                for k in rt_node_dic[rt_node[i + 1]]:
                    edge.append([j, k, 0])

    return num_node + 2, edge


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = {}

    def AddEdge(self, edge):
        if edge[0] not in self.graph.keys():
            self.graph[edge[0]] = [(edge[1], edge[2])]
        else:
            self.graph[edge[0]].append((edge[1], edge[2]))

    def TopologicalSort(self, v, visited, stack):
        visited[v] = True

        if v in self.graph.keys():
            for node, weight in self.graph[v]:
                if visited[node] == False:
                    self.TopologicalSort(node, visited, stack)

        stack.append(v)

    def ShortestPath(self, s, t):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.TopologicalSort(s, visited, stack)

        dist = [1] * (self.V)
        dist[s] = 0
        ancestor = [None] * (self.V)

        while stack:
            i = stack.pop()
            if i in self.graph.keys():
                for node, weight in self.graph[i]:
                    if dist[node] > dist[i] + weight:
                        dist[node] = dist[i] + weight
                        ancestor[node] = i
        return dist[t], ancestor


def PathExtraction(ancestors):
    idx = ancestors[-1]
    path_tmp = [idx]
    while idx is not None and idx != 0:
        path_tmp.append(ancestors[idx])
        idx = ancestors[idx]
    path_tmp.reverse()
    return path_tmp


def PathRecoverToRT(path_node, node_rt_dic, num_node):
    path_rt = []
    path_mz = []
    path_charge = []
    for i in path_node:
        if i != num_node - 1 and i != 0:
            path_rt.append(node_rt_dic[i][0])
            path_mz.append(node_rt_dic[i][1])
            path_charge.append(node_rt_dic[i][2])
    return path_rt, path_mz, path_charge


def RemoveVisited(path_node):
    index = []
    for i in range(len(path_node)):
        if i != 0 and path_node[i] % 2 == 0 and path_node[i - 1] + 1 == path_node[i]:
            index.append(path_node[i] // 2 - 1)
    return index


def PathGen(data, intensity_accu, num_path, delay, min_time, max_time):
    paths_rt = []
    paths_mz = []
    paths_charge = []
    lengths = []
    _, _, _, _, edge_intensity_dic = NodeEdge1Create(data, intensity_accu, delay, min_time, max_time)
    for i in range(num_path):
        num_node, rt_node_dic, node_rt_dic, edge, _ = NodeEdge1Create(
            data, intensity_accu, delay, min_time, max_time
        )
        num_node, edge = Edge0Create(num_node, rt_node_dic, edge)
        g = Graph(num_node)
        for j in edge:
            g.AddEdge(j)
        s = 0
        t = num_node - 1
        length, ancestors = g.ShortestPath(s, t)
        if length >= 0:
            break
        logger.info('[%d/%d]: features: %d, rest: %d' %(i+1,num_path,-length, len(data)))
        lengths.append(-length)
        path_node = PathExtraction(ancestors)
        path_rt, path_mz, path_charge = PathRecoverToRT(
            path_node, node_rt_dic, num_node
        )
        paths_rt.append(path_rt)
        paths_mz.append(path_mz)
        paths_charge.append(path_charge)
        data = np.delete(data, RemoveVisited(path_node), axis=0)

    return paths_rt, paths_mz, paths_charge, edge_intensity_dic


def WriteFile(
    outfile_name, paths_rt, paths_mz, paths_charge, edge_intensity_dic, isolation, delay, min_time, max_time
):
    if len(paths_rt) != len(paths_mz):
        logger.error("length of rt and mz are not the same: rt %d, mz %d", len(paths_rt), len(paths_mz))
        return
    text_file = open(outfile_name, "wt")
    for i in range(len(paths_rt)):
        n = text_file.write("path" + str(i) + "\t")
        if len(paths_rt[i]) != len(paths_mz[i]):
            logger.error("length of rt and mz are not the same: rt %d, mz %d", len(paths_rt[i]), len(paths_mz[i]))
            break
        for j in range(len(paths_rt[i])):
            mz_index = paths_mz[i][j]
            iso = isolation
            start = paths_rt[i][j]
            charge = paths_charge[i][j]
            if j != len(paths_rt[i]) - 1:
                stop = paths_rt[i][j + 1]
                dur = stop - start - delay
                intensity = 0
                mid = (stop + start) / 2.0
                if (start, stop) in edge_intensity_dic.keys():
                    intensity = edge_intensity_dic[(start, stop)]
                    if dur < min_time - 1e-4 or dur > max_time + 1e-4:
                        logging.error("dur should not be < min scan time or > max scan time: %.4f", dur)
                    n = text_file.write(
                        "{:.4f}".format(mz_index)
                        + " "
                        + "{:.4f}".format(iso)
                        + " "
                        + "{:.4f}".format(dur)
                        + " "
                        + "{:.4f}".format(start)
                        + " "
                        + "{:.4f}".format(stop - delay)
                        + " "
                        + "{:.4f}".format(intensity)
                        + " "
                        + "{:.4f}".format(mid)
                        + " "
                        + str(charge)
                        + "\t"
                    )
        n = text_file.write("\n")
    text_file.close()
