import os
import subprocess
import re

class MWIS():
    """Class for solving weighted maximum independent sets.
    Using methods in paper: Optimisation of unweighted/weighted maximum independent sets and minimum vertex covers
    github: https://github.com/darrenstrash/open-pls.git
    """

    def __init__(self, name, path=os.path.join('inputs', 'graph')):
        self.name = name
        self.graph_file = os.path.join(path, self.name+'.graph')
        self.weight_file = os.path.join(path, self.name+'.graph.weights')
        self.out_file = os.path.join(path, self.name+'.sol')

    def local_search(self):
        """Find weighted maximum independent sets in given graph
        """
        command = './open-pls-1.0/bin/pls --algorithm=mwis --input-file=' + self.graph_file + ' --weighted --use-weight-file --timeout=1 --random-seed=0 > ' + self.out_file
        # os.system(command)
        process = subprocess.Popen(command.split())
        process.wait()
        # read results
        sol_pattern = re.compile(r"best-solution   :(( \d+)+)")
        with open(self.out_file, 'r') as f:
            content = f.read()
        sol_match = re.search(sol_pattern, content)
        if sol_match:
            node_list = [int(x) for x in sol_match.group(1).split()]
            return node_list
        else:
            print("ERROR!: no mwis solution found")
            return -1

    def write_graph(self, str_list, str_weights):
        """Write graph into file
        """
        with open(self.graph_file, 'w') as f:
            f.writelines(str_list)
        with open(self.weight_file, 'w') as f:
            f.writelines(str_weights)

    def treeToGraph(self, T, trackTrees, treeNo, timeConflict=False):
        """chnage all the tree in trackTrees in a Graph for MWIS.
        inputs:
            T: total time in the target video
        """
        paths = []
        for treeId in range(len(trackTrees[treeNo])):
            # get all track path in a tree
            tracks = trackTrees[treeNo][treeId].paths_to_leaves()
            for track in tracks:
                # track: list of node identifier from root to leaf
                # change each track with identifier to number such as 0013430012
                # 0 represent the absent of the frame. detection start from 1
                # score updating
                track_list = [0]*T
                for node in track:
                    index = int(node.split('_')[3]) + 1  # index should from 1, 0 means missing
                    time = int(node.split('_')[1])
                    track_list[time] = index
                # score
                score = trackTrees[treeNo][treeId].nodes[track[-1]].data['score']
                paths.append({'treeId': treeId, 'obj_id':treeNo, 'track': track, 'track_list': track_list, 'weight': score})
        # judge whether there is an edge between two node
        def ifConnectedInRoi(node1, node2):
            for trackId in range(len(node1['track_list'])):
                if node1['track_list'][trackId] == 0 or node2['track_list'][trackId] == 0:
                    continue
                if node1['track_list'][trackId] == node2['track_list'][trackId]:
                    return True
            return False
        def ifConnectedInTime(node1, node2):
            for trackId in range(len(node1['track_list'])):
                if node1['track_list'][trackId] != 0 and node2['track_list'][trackId] != 0 and node1['track_list'][trackId] != node2['track_list'][trackId]:
                    return True
            return False
        ifConnected = ifConnectedInTime if timeConflict else ifConnectedInRoi
        # get all tracks, now get edges
        edges = []
        for i in range(len(paths)-1):
            for j in range(i+1, len(paths)):
                # judge two node
                if ifConnected(paths[i], paths[j]):
                    edges.append((i,j))
        # get weight for each track
        # TODO: for each track, we get a score
        weights = []
        for path in paths:
            print(path['weight'])
            weights.append(path['weight'])
        # write graph
        graph_dict = {}
        print(paths)
        print(edges)
        for edge in edges:
            if edge[0] in graph_dict:
                graph_dict[edge[0]].append(edge[1])
            else:
                graph_dict[edge[0]] = [edge[1]]
            if edge[1] in graph_dict:
                graph_dict[edge[1]].append(edge[0])
            else:
                graph_dict[edge[1]] = [edge[0]]
        # write graph file
        str_list = ['%d %d\n'%(len(paths), len(edges))]
        has_graph = False
        for i in range(len(paths)):
            if i in graph_dict:
                # notice that the edge of mwis is from 1
                str_list.append(' '.join(str(e+1) for e in graph_dict[i])+'\n')
                has_graph = True
            else:
                str_list.append('\n')
        if not has_graph:
            return [], []
        # write weight file
        str_weights = ['%d %f\n'%(i, weights[i]) for i in range(len(weights))]
        self.write_graph(str_list, str_weights)
        # find best solution
        results = self.local_search()
        assert results != -1
        print('best solation found--------------')
        print(results)
        return paths, results

def main():
    mwis = MWIS('test-my')
    results = mwis.local_search()
    print(results)

if __name__ == '__main__':
    main()