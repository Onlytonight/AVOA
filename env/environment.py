import gc

from util.functions import pairwise_iteration

import networkx as nx
import numpy as np
import copy
import os
import math


DEFAULT_EDGE_ATTRIBUTES = {
    'increments': 1,
    'reductions': 1,
    'weight': 0.0,
    'traffic': 0.0
}


class Environment(object):

    def __init__(self,
                 env_type='GBN',
                 traffic_profile='uniform',
                 routing='ecmp',
                 action_type='all_link_weight',
                 reward_magnitude='link_utilization',
                 base_reward='diff_min_max',
                 reward_computation='value',
                 base_dir='datasets',
                 result_dir='result',
                 init_sample=0,
                 seed_init_weights=1,
                 min_weight=1.0,
                 max_weight=4.0,
                 weigths_to_states=False,
                 link_traffic_to_states=False,
                 link_utilization_to_states=True,
                 probs_to_states=False,
                 available_bw=True,
                 K=6
                 ):
        '''
        :param env_type: 网络名称
        :param traffic_profile: 使用的流量
        :param routing: 路由方式  sp最短路径 k-sp为k条最短路径  ecmp分支路由
        :param action_type: all_link_weight host-host
        :param reward_magnitude:
        :param base_reward:
        :param reward_computation:
        :param base_dir:
        :param result_dir:
        :param init_sample:
        :param seed_init_weights:
        :param min_weight:
        :param max_weight:
        :param weigths_to_states:
        :param link_traffic_to_states:
        :param link_utilization_to_states:
        :param probs_to_states:
        :param K: ksp或 host-host模式时的K条路径
        '''
        env_type = [env for env in env_type.split('+')]#env_type if type(env_type) == list else [env_type]
        self.env_type = env_type
        self.traffic_profile = traffic_profile
        self.routing = routing
        self.init_sample=init_sample
        self.num_sample = init_sample
        self.seed_init_weights = seed_init_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.action_type=action_type

        num_features = 0
        self.weigths_to_states = weigths_to_states
        self.state_type='state'
        if self.weigths_to_states:
            num_features += 1
            self.state_type+='_weights'
        self.link_traffic_to_states=link_traffic_to_states
        if self.link_traffic_to_states :
            num_features+=1
            self.state_type+='_traffic'
        self.link_utilization_to_states = link_utilization_to_states
        if self.link_utilization_to_states:
            num_features += 1
            self.state_type+='_utilization'
        self.probs_to_states = probs_to_states
        if self.probs_to_states:
            num_features += 2
            self.state_type+='_probs'
        self.num_features = num_features
        self.available_bw=available_bw
        if self.available_bw:
            num_features += 1
            self.state_type+='_availableBW'


        self.reward_magnitude = reward_magnitude
        self.base_reward = base_reward
        self.reward_computation = reward_computation

        self.base_dir = base_dir
        self.dataset_dirs = []
        for env in env_type:
            self.dataset_dirs.append(os.path.join(base_dir, env, traffic_profile))

        self.result_dir=result_dir

        self.graph_data=dict()
        self.K=K


        self.initialize_environment()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        self.reward_measure = self.compute_reward_measure()
        self.set_target_measure()





    def initialize_environment(self, num_sample=None, random_env=False):
        '''
        初始化环境：加载拓扑；生成网络图；加载链路带宽容量；加载流量矩阵
        :param num_sample:
        :param random_env: 随机环境，在self.env_type里随机
        :return:
        '''
        if num_sample is not None:
            self.num_sample = num_sample
        else:
            self.num_sample += 1
        if random_env:
            num_env = np.random.randint(0,len(self.env_type))
        else:
            num_env = self.num_sample % len(self.env_type)
        self.network = self.env_type[num_env]
        self.dataset_dir = self.dataset_dirs[num_env]

        self._load_topology_object()
        self.generate_graph()
        self._load_capacities()
        self._load_traffic_matrix()
        if self.action_type=='host-host':
            self._num_shortest_path()
            print(self.allPaths)

    def reset(self, change_sample=False):
        '''
        重设环境的函数
        :param change_sample: 重设时是否切换流量
        '''
        if change_sample:
            self.next_sample()
        else:
            if self.seed_init_weights is None: self._define_init_weights()

            self._reset_edge_attributes()

        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        self.reward_measure = self.compute_reward_measure()
        self.set_target_measure()

    def step(self, action, step_back=False):

        if self.action_type=='all_link_weight':
            #action 为全局链路权重  [link_num]  对应链路权重方法
            self.push_weights_G(action)
            self.weights=copy.deepcopy(action)
            self.get_weights()
            self._generate_routing()
            self._get_link_traffic()
            state = self.get_state()
            reward = self._compute_reward()
            return state, reward

        elif self.action_type=='host-host':
            # action 为路由，即先用SP算法生成每个主机间的基于  跳数\带宽 的 K 条路径， action则为每隔主机对选择一条路径
            # 根据动作选择路由
            routing=self._host_host_action(action)
            print(routing)
            self._generate_routing()
            self._get_link_traffic(routing)
            state = self.get_state()
            reward = self._compute_reward()
            return state, reward

        elif self.action_type=='rsir':
            self._generate_routing()
            self._get_link_traffic(action)
            # state = self.get_state(normalized=False)
            reward = self._compute_reward()
            return reward


    def get_env_info(self):
        '''
        被外部调用
        :return:
            环境信息：包括使用环境、流量类型、流量是否变化、动作类型、路由方式、奖励
        '''
        env_info=self.network + '-' + \
                 self.traffic_profile + '-' + \
                 self.action_type + '-' + \
                 self.routing + '-' + \
                 self.state_type + '-' + 'init_sample_'+str(self.init_sample)
        link_utilization = copy.deepcopy(self.link_utilization)
        link_traffic = copy.deepcopy(self.link_traffic)
        return env_info,link_traffic,link_utilization,self.num_sample


    def next_sample(self):
        if len(self.env_type) > 1:
            self.initialize_environment()
        else:
            self.num_sample += 1
            self._reset_edge_attributes()
            self._load_capacities()
            self._load_traffic_matrix()

    def define_num_sample(self, num_sample):
        self.num_sample = num_sample - 1

    def generate_graph(self):
        G  = copy.deepcopy(self.topology_object)
        self.n_nodes = G.number_of_nodes()
        self.n_links = G.number_of_edges()
        self._define_init_weights()
        idx = 0
        link_ids_dict = {}
        for (i,j) in G.edges():
            G[i][j]['id'] = idx
            G[i][j]['increments'] = 1
            G[i][j]['reductions'] = 1
            G[i][j]['weight'] = copy.deepcopy(self.init_weights[idx])
            link_ids_dict[idx] = (i,j)
            G[i][j]['capacity'] = G[i][j]['bandwidth']
            G[i][j]['traffic'] = 0.0
            G[i][j]['utilization'] = 0.0
            G[i][j]['available_bw'] = 0.0
            idx += 1
        self.G = G
        incoming_links, outcoming_links = self._generate_link_indices_and_adjacencies()
        self.graph_data['link_ids_dict'] = link_ids_dict
        self.graph_data['incoming_links'] = incoming_links
        self.graph_data['outcoming_links'] = outcoming_links

    def get_edge_index(self):
        '''
        :return: 返回一个list   关于边连接的的list  相当于把边视为节点，节点视为边  形如[[1,2],[1,3],[1,5]...]
        '''
        # return [[self.graph_data['incoming_links'][i],self.graph_data['outcoming_links'][i]] for i in range(len(self.graph_data['incoming_links']))  ]

        return (self.graph_data['incoming_links'], self.graph_data['outcoming_links'])

    def get_adj_matrix(self):
        return nx.adjacency_matrix(self.G).todense()


    def set_target_measure(self):
        self.target_sp_routing = copy.deepcopy(self.sp_routing)
        self.target_reward_measure = copy.deepcopy(self.reward_measure)
        self.target_link_traffic = copy.deepcopy(self.link_utilization)
        self.get_weights()
        self.target_weights = copy.deepcopy(self.raw_weights)


    def get_weights(self):
        weights = [0.0]*self.n_links
        for i,j in self.G.edges():
            weights[self.G[i][j]['id']] = copy.deepcopy(self.G[i][j]['weight'])
        self.raw_weights = weights
        max_weight = self.max_weight*3
        # 归一化处理
        self.weights = [weight/max_weight for weight in weights]
        # print('weights:',self.weights,np.shape(self.weights))

    def get_state(self,normalized=True):
        state = []

        link_utilization = copy.deepcopy(self.link_utilization)
        link_traffic = copy.deepcopy(self.link_traffic)
        weights = copy.deepcopy(self.weights)
        p_in=copy.deepcopy(self.p_in)
        p_out=copy.deepcopy(self.p_out)
        link_available_bw=copy.deepcopy(self.link_available_bw)

        if self.link_utilization_to_states:
            if normalized:
                link_utilization = (link_utilization - np.min(link_utilization)) / (np.max(link_utilization) - np.min(link_utilization))
            state.extend(link_utilization)
        if self.link_traffic_to_states:
            if normalized:
                link_traffic= (link_traffic - np.min(link_traffic)) / (np.max(link_traffic) - np.min(link_traffic))
            state.extend(link_traffic)
        if self.weigths_to_states:
            if normalized:
                weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            state.extend(weights)
        if self.probs_to_states:
            if normalized:
                p_in = (p_in - np.min(p_in)) / (np.max(p_in) - np.min(p_in))
                p_out = (p_out - np.min(p_out)) / (np.max(p_out) - np.min(p_out))
            state.extend(p_in)
            state.extend(p_out)
        if self.available_bw:
            if normalized:
                link_available_bw = (link_available_bw - np.min(link_available_bw)) / (np.max(link_available_bw) - np.min(link_available_bw))
            state.extend(link_available_bw)

        return np.array(state, dtype=np.float32)

    def define_weight(self, link, weight):
        i, j = link
        self.G[i][j]['weight'] = weight
        self._generate_routing()
        self._get_link_traffic()

    def push_weights_G(self,action):
        for l in range(self.n_links):
            link=self.graph_data['link_ids_dict'][l]
            i, j = link
            self.G[i][j]['weight'] = action[l]




    def reinitialize_weights(self, seed_init_weights=-1, min_weight=None, max_weight=None):
        if seed_init_weights != -1:
            self.seed_init_weights = seed_init_weights
        if min_weight: self.min_weight = min_weight
        if max_weight: self.max_weight = max_weight

        self.generate_graph()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()

    def reinitialize_routing(self, routing):
        self.routing = routing
        self._get_link_traffic()




    # in the q_function we want to use info on the complete path (src_node, next_hop, n3, n4, ..., dst_node)
    # this function returns the indices of links in the path
    def get_complete_link_path(self, node_path):
        link_path = []
        for i, j in pairwise_iteration(node_path):
            link_path.append(self.G[i][j]['id'])
        # pad the path until "max_length" (implementation is easier if all paths have same size)
        link_path = link_path + ([-1] * (self.n_links-len(link_path)))
        return link_path



    """
    ****************************************************************************
                 PRIVATE FUNCTIONS OF THE ENVIRONMENT CLASS
    ****************************************************************************
    """

    def _num_shortest_path(self):
        '''
        生成K条基于跳数的最短路径
        Returns:

        '''
        self.G_ud=self.G.to_undirected()
        print(nx.is_connected(self.G_ud))
        print(self.G_ud.nodes)
        print(self.G_ud.edges)
        self.diameter = nx.diameter(self.G_ud)
        self.allPaths={}
        # Iterate over all node1,node2 pairs from the graph
        for n1 in self.G_ud:
            for n2 in self.G_ud:
                if (n1 != n2):
                    # Check if we added the element of the matrix
                    if str(n1) + ':' + str(n2) not in self.allPaths:
                        self.allPaths[str(n1) + ':' + str(n2)] = []

                    # First we compute the shortest paths taking into account the diameter
                    # This is because large topologies might take too long to compute all shortest paths
                    [self.allPaths[str(n1) + ':' + str(n2)].append(p) for p in
                     nx.all_simple_paths(self.G_ud, source=n1, target=n2, cutoff=self.diameter * 2)]

                    # We take all the paths from n1 to n2 and we order them according to the path length
                    self.allPaths[str(n1) + ':' + str(n2)] = sorted(self.allPaths[str(n1) + ':' + str(n2)],
                                                                    key=lambda item: (len(item), item))

                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1) + ':' + str(n2)]):
                        currentPath = self.allPaths[str(n1) + ':' + str(n2)][path]
                        i = 0
                        j = 1

                        path = path + 1

                    # Remove paths not needed
                    del self.allPaths[str(n1) + ':' + str(n2)][path:len(self.allPaths[str(n1) + ':' + str(n2)])]
                    gc.collect()


    # reward function is currently quite simple
    def compute_reward_measure(self, measure=None):
        if measure is None:
            if self.reward_magnitude == 'link_utilization':
                measure = self.link_utilization
            elif self.reward_magnitude == 'link_traffic':
                measure = self.link_traffic
            elif self.reward_magnitude == 'weights':
                measure = self.raw_weights

        if self.base_reward == 'mean_times_std':
            return np.mean(measure) * np.std(measure)
        elif self.base_reward == 'mean':
            return np.mean(measure)
        elif self.base_reward == 'std':
            return np.std(measure)
        elif self.base_reward == 'diff_min_max':
            return np.max(measure) - np.min(measure)
        elif self.base_reward == 'min_max':
            return np.max(measure)

    #
    def _compute_reward(self):
        congestion = sum(i >= 1 for i in self.link_utilization) / len(self.link_utilization)
        current_reward_measure = self.compute_reward_measure()

        if self.reward_computation == 'value':
            reward = 5 - math.log(math.pow(current_reward_measure+ 0.1, 19))-math.pow(congestion, 2)*200
            # print('current_reward_measure',current_reward_measure)
        elif self.reward_computation == 'change':
            reward = self.reward_measure - current_reward_measure

        self.reward_measure = current_reward_measure
        #
        # if congestion>0.1:
        #     reward=-congestion*10
        return reward


    def _load_topology_object(self):
        try:
            nx_file = os.path.join(self.base_dir, self.network, 'graph_attr.txt')  # 'datasets\\NSFNet\\graph_attr.txt'
            self.topology_object = nx.DiGraph(nx.read_gml(nx_file, destringizer=int))
        except:
            self.topology_object = nx.DiGraph()
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph.txt')
            print(capacity_file)
            with open(capacity_file) as fd:
                for line in fd:
                    if 'Link_' in line:
                        camps = line.split(" ")
                        self.topology_object.add_edge(int(camps[1]),int(camps[2]))
                        self.topology_object[int(camps[1])][int(camps[2])]['bandwidth'] = int(camps[4])

    def _load_capacities(self):
        if self.traffic_profile == 'gravity_full':
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph-TM-'+str(self.num_sample)+'.txt')
        else:
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph.txt')
        with open(capacity_file) as fd:
            for line in fd:
                if 'Link_' in line:
                    camps = line.split(" ")
                    self.G[int(camps[1])][int(camps[2])]['capacity'] = int(camps[4])

    def _load_traffic_matrix(self):
        if self.num_sample>199:
            self.num_sample = 0
        # print('TM-', self.num_sample)
        tm_file = os.path.join(self.dataset_dir, 'TM', 'TM-'+str(self.num_sample))
        self.traffic_demand = np.zeros((self.n_nodes,self.n_nodes))
        with open(tm_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                self.traffic_demand[int(camps[1]),int(camps[2])] = float(camps[3])
        self.get_link_probs()

    def _define_init_weights(self,random_weights=True):
        if random_weights:
            np.random.seed(seed=self.seed_init_weights)
            self.init_weights = np.random.randint(self.min_weight,self.max_weight+1,self.n_links)
            np.random.seed(seed=None)
        else:
            self.init_weights = np.ones(self.n_links)

    # generates indices for links in the network
    def _generate_link_indices_and_adjacencies(self):
        '''
        for the q_function, we want to have info on link-link connection points
        there is a link-link connection between link A and link B if link A
        is an incoming link of node C and link B is an outcoming node of node C.
        For connection "i", the incoming link is incoming_links[i] and the
        outcoming link is outcoming_links[i]
        '''
        incoming_links = []
        outcoming_links = []
        # iterate through all links
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                incoming_link_id = self.G[i][j]['id']
                # for each link, search its outcoming links
                for k in self.G.neighbors(j):
                    outcoming_link_id = self.G[j][k]['id']
                    incoming_links.append(incoming_link_id)
                    outcoming_links.append(outcoming_link_id)

        return incoming_links, outcoming_links

    def _reset_edge_attributes(self, attributes=None):
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list: attributes = [attributes]
        for (i,j) in self.G.edges():
            for attribute in attributes:
                if attribute == 'weight':
                    self.G[i][j][attribute] = copy.deepcopy(self.init_weights[self.G[i][j]['id']])
                else:
                    self.G[i][j][attribute] = copy.deepcopy(DEFAULT_EDGE_ATTRIBUTES[attribute])

    def _normalize_traffic(self):
        '''
        实际上计算链路利用率，并把高出链路带宽的部分裁掉，链路利用率最高为1
        :return:
        '''
        for (i,j) in self.G.edges():

            if self.G[i][j]['traffic']>self.G[i][j]['capacity']:
                self.G[i][j]['traffic']=self.G[i][j]['capacity']
            self.G[i][j]['utilization'] = self.G[i][j]['traffic'] / self.G[i][j]['capacity']
            self.G[i][j]['available_bw'] = self.G[i][j]['capacity'] - self.G[i][j]['traffic']

    def _generate_routing(self, next_hop=None):
        self.sp_routing = dict(nx.all_pairs_dijkstra_path(self.G))
        # print(self.sp_routing)
        # deg = []
        # for i in self.G.nodes():
        #     deg.append(self.G.degree(i)//2)
        # print(deg)
        #self.path_lengths = dict(nx.all_pairs_dijkstra_path_length(self.G))

    def _host_host_action(self,action):
        routing=dict()
        for i in range(np.shape(action)[0]):
            routing_t = dict()
            for j in range(np.shape(action)[0]):
                if i == j:
                    routing_t[j]=[j]
                else:
                    routing_t[j] = self.allPaths[str(i)+':'+str(j)][int(action[i][j])]
            routing[i]=routing_t

        return routing


    # returns a list of traffic volumes of each link
    def _distribute_link_traffic(self, routing=None):
        self._reset_edge_attributes('traffic')
        if self.action_type!='host-host':
            if self.routing == 'sp':
                if routing is None: routing = self.sp_routing
                for i in self.G.nodes():
                    for j in self.G.nodes():
                        if i == j: continue
                        traffic = self.traffic_demand[i][j]
                        for u,v in pairwise_iteration(routing[i][j]):
                            self.G[u][v]['traffic'] += traffic
            elif self.routing == 'k-sp':

                pass
            elif self.routing == 'ecmp':
                visited_pairs = set()
                self.next_hop_dict = {i : {j : set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())}
                for src in range(self.G.number_of_nodes()-1):
                    for dst in range(self.G.number_of_nodes()-1):
                        if src == dst: continue
                        if (src,dst) not in visited_pairs:
                            routings = set([item for sublist in
                                            [[(routing[i], routing[i + 1]) for i in range(len(routing) - 1)] for routing in
                                             nx.all_shortest_paths(self.G, src, dst, 'weight')] for item in sublist])
                            # for routing in nx.all_shortest_paths(self.G, src, dst, 'weight'):
                            #     for i in range(len(routing-1)):
                            #         for sublist in

                            for (new_src,next_hop) in routings:
                                self.next_hop_dict[new_src][dst].add(next_hop)
                                visited_pairs.add((new_src,dst))
                        traffic = self.traffic_demand[src][dst]

                        self.successive_equal_cost_multipaths(src, dst, traffic)
        elif self.action_type=='host-host':
            if routing is None: routing = self.sp_routing
            for i in self.G.nodes():
                for j in self.G.nodes():
                    if i == j: continue
                    traffic = self.traffic_demand[i][j]
                    for u, v in pairwise_iteration(routing[i][j]):
                        self.G[u][v]['traffic'] += traffic

    def successive_equal_cost_multipaths(self, src, dst, traffic):
        new_srcs = self.next_hop_dict[src][dst]
        traffic /= len(new_srcs)
        for new_src in new_srcs:
            self.G[src][new_src]['traffic'] += traffic
            if new_src != dst:
                self.successive_equal_cost_multipaths(new_src, dst, traffic)

    def _get_link_traffic(self, routing=None):

        self._distribute_link_traffic(routing)
        self._normalize_traffic()
        link_traffic = [0]*self.n_links
        for i,j in self.G.edges():
            link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
        self.link_traffic = link_traffic
        # self.mean_traffic = np.mean(link_traffic)

        link_utilization = [0]*self.n_links
        for i,j in self.G.edges():
            link_utilization[self.G[i][j]['id']] = self.G[i][j]['utilization']
        self.link_utilization = link_utilization



        link_available_bw = [0]*self.n_links
        for i,j in self.G.edges():
            link_available_bw[self.G[i][j]['id']] = self.G[i][j]['available_bw']
        self.link_available_bw = link_available_bw

        # self.mean_utilization = np.mean(link_utilization)
        # self.get_weights()

    def get_link_traffic(self):
        link_traffic = [0]*self.n_links
        for i,j in self.G.edges():
            link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
        return link_traffic

    def get_link_probs(self):
        traffic_in = np.sum(self.traffic_demand, axis=0)
        traffic_out = np.sum(self.traffic_demand, axis=1)
        node_p_in = list(traffic_in / np.sum(traffic_in))
        node_p_out = list(traffic_out / np.sum(traffic_out))
        self.p_in = [0]*self.n_links
        self.p_out = [0]*self.n_links
        for i,j in self.G.edges():
            self.p_in[self.G[i][j]['id']] = node_p_out[i]
            self.p_out[self.G[i][j]['id']] = node_p_in[j]


    def betweeness(self):
        G = self.G
        all_shortest_path = []
        result_dict = {}
        node_list = list(G.nodes)
        for i in range(len(node_list)):
            for j in range(i+1,len(node_list)):
                u = node_list[i]
                v = node_list[j]
                all_shortest_path += list(nx.all_shortest_paths(G, u, v))
        all_shortest_path_num = len(all_shortest_path)
        for edge in G.edges:
            u, v = edge
            result_dict[(u,v)] = 0
            for path in all_shortest_path:
                if u in path and v in path:
                    result_dict[(u,v)] += 1
            result_dict[(u,v)] /= all_shortest_path_num
        between = np.zeros((len(self.G.edges)))
        for i,j in self.G.edges():
            between[self.G[i][j]['id']] = result_dict[(i, j)]
        return between

    def bandwidth(self):
        bd = np.zeros((len(self.G.edges)))
        for i, j in self.G.edges():
            bd[self.G[i][j]['id']] = self.G[i][j]['bandwidth']
        return bd