"""
Classes for handling Graphs for single clip and full movie.
"""

import re
import json
import warnings
import itertools
import numpy as np
import networkx as nx
from collections import OrderedDict, defaultdict

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# Local imports
# import data_loaders
# import lemmatizer


def load_movie_graph(movie, users, overwrite=False):
    """Load a movie graph for given movie.
    """

    data_loaders.copy_latest_annots(movie, users, overwrite)
    castlist = data_loaders.load_castlist(movie)
    annot_list = data_loaders.list_annots(movie)

    # load individual JSON files and add clip-graphs to movie-graph
    movie_graph = MovieGraph(movie, castlist)
    for sid, fname in annot_list.iteritems():
        try:
            with open(fname, 'r') as fid:
                graph_json = json.load(fid)
            clip_graph = ClipGraph(graph_json)
            clip_graph.add_chid_to_entities(movie_graph.castlist)
            movie_graph.add_clip_graph(sid, clip_graph)
        except ValueError:
            warnings.warn('Failed to load or create CG from json file: %s' %fname)

    return movie_graph


def get_relationship_directions():
    """Get the associations between relationships and directions.
    """
    rel_directions = {}
    for reln in data_loaders.VOCAB['relationships']:
        for r in reln['values']:
            if '(directed)' in r['description']:
                rel_directions.update({r['value']: 'directed'})
            elif '(undirected)' in r['description']:  #TODO: equivalent to bidirected?
                rel_directions.update({r['value']: 'undirected'})


class ClipGraph(object):
    """A graph representing the story for one video clip.
    """

    def __init__(self, graph_json):
        """Create a graph from the original JSON annotation dumps.
        """

        self.orig_graph_json = graph_json
        # general clip information
        self.situation = graph_json['situation']
        self.scene_label = graph_json['scene']
        self.description = graph_json['sentence_description']
        # things that should be done for every clip graph
        self.update_video_information([graph_json['video']])
        self.convert_to_nx_graph()
        # self.resolve_edges(get_relationship_directions())
        # self.lemmatize()

    def update_video_information(self, video_fnames):
        """Associate clip graph with video information.
            video_fname (can be a list of multiple filenames)
        NOTE: scene, start/end-shot are all indexed by 1.
        """

        self.video = {'movie': '', 'fname': [], 'scene': [], 'ss': 9999, 'es': -1}
        for vf in video_fnames:
            _, movie, fname = vf.rsplit('/', 2)
            sc, ss, es = [int(info.split('-')[1]) for info in fname.split('.')[0:3]]
            self.video['fname'].append(fname)
            self.video['scene'].append(sc)
            if ss < self.video['ss']:   self.video['ss'] = ss
            if es > self.video['es']:   self.video['es'] = es
        self.video['movie'] = movie

    def convert_to_nx_graph(self, castlist=None):
        """Convert the JSON dump to a NetworkX graph class.
            TODO: "reason" nodes are currently ignored
        """

        G = nx.DiGraph()
        ### add nodes to the graph
        for node in self.orig_graph_json['nodes']:
            # TODO: check node is not empty or invalid?
            if not node['name'].strip():
                continue

            if node['id'] in G.nodes():
                print("BROKEN: Found duplicate node-id {} in graph for {}".format(node['id'], self.video['fname']))
                continue
                # raise RuntimeError('Node with same id already exists!')

            # add entity nodes
            if node['type'] == 'entity':
                G.add_node(node['id'], name=node['name'], node_id=node['node_id'])

            # add attribute nodes
            elif node['type'] == 'attribute':
                subtype = ''          # default ''
                text = node['name']   # default "name"
                if ':' in node['name']:
                    subtype, text = node['name'].split(':')
                G.add_node(node['id'], name=text, subtype=subtype)

            elif node['type'] == 'time':
                if 't_start' in node and 't_end' in node:
                    G.add_node(node['id'], name=node['name'], start=node['t_start'], end=node['t_end'])
                else:
                    warnings.warn("Time node without proper assignment!", RuntimeWarning)
                    G.add_node(node['id'], name=node['name'])

            # add all other nodes
            else:
                G.add_node(node['id'], name=node['name'])

            # add position and type
            # G.add_node(node['id'], origtext=node['name'], type=node['type'], pos=(node['x'], node['y']))
            if 'x' in node and 'y' in node:
                G.add_node(node['id'], origtext=node['name'], type=node['type'], pos=(node['x'], node['y']))
            else:
                G.add_node(node['id'], origtext=node['name'], type=node['type'], pos=(0, 0))

        if self.situation:
            G.add_node(-1, origtext=self.situation,   name=self.situation,   type='situation', pos=(0, 0))
        if self.scene_label:
            G.add_node(-2, origtext=self.scene_label, name=self.scene_label, type='scene', pos=(0, 10))

        ### add edges to the graph
        for edge in self.orig_graph_json['edges']:
            # check nodes exist
            if edge['source'] in G.nodes() and edge['target'] in G.nodes():
                G.add_edge(edge['source'], edge['target'])
            else:
                warnings.warn('Edge source/target node not in graph. %d --> %d' \
                            %(edge['source'], edge['target']), RuntimeWarning)

        # save
        self.G = G

    def add_chid_to_entities(self, castlist):
        """Map node_ids to castlist chid.
        """
        for n in self.G.nodes():
            if self.node_type(n, 'entity'):
                node_id = self.G.node[n]['node_id']
                if node_id < len(castlist):
                    self.G.add_node(n, chid=castlist[node_id]['chid'])
                else:  # UNLISTED CHARACTERS
                    self.G.add_node(n, chid='---')

    def check_chid_mappings(self, castlist):
        """Print the chid mapping information to see whether it's ok!
        """
        for n in self.G.nodes():
            if self.node_type(n, 'entity'):
                chid = self.G.node[n]['chid']
                castlist_name = [c['name'] for c in castlist if c['chid'] == chid]
                print('{:4d} | {:40s} | {:15s} | {:40s}'.format(n, self.G.node[n]['name'], chid, castlist_name))

    def node_name(self, n, verify_name=None):
        """Returns or verifies the name of a node n.
            if verify_name is given, returns True/False
            if verify_name is None,  returns name of the node
        """

        if verify_name:
            verify_name = verify_name.strip()
            # check both sub-type and name if attribute
            if self.G.node[n]['type'] == 'attribute':
                return self.G.node[n]['subtype'] + ':' + self.node_name(n) == verify_name
            else:
                return self.node_name(n) == verify_name

        # get the node name
        return self.G.node[n]['name'].strip()

    def node_type(self, n, ntype=None):
        """Returns or verifies the node type of node n.
            if ntype is given, returns True/False
            if ntype is None,  returns actual type
        """

        if ntype:
            if n in self.G.nodes() and ntype == self.G.node[n]['type']:
                return True
            else:
                return False
        else:
            if n in self.G.nodes():
                return self.G.node[n]['type']
            else:
                return None

    def get_nodes_of_type(self, ntype):
        """Returns a list of names of all nodes of type ntype.
        """

        return [self.G.node[n]['name'] for n in self.G.nodes() if self.G.node[n]['type'] == ntype]

    def get_node_ids_of_type(self, ntypes=None):
        """Returns the node ids of all nodes that have a type in the ntypes list.
        """
        return [nid for nid in self.G.nodes() if self.node_type(nid) in ntypes]

    def get_node_type_dict(self, ntypes=None):
        """Returns a dictionary mapping each node type in ntypes to a list of node names.

        Example: { 'entity': ['John', 'Ashley'], 'attribute': ['shy', 'happy']}
        """

        type_dict = defaultdict(list)
        if not ntypes:
            ntypes = [self.G.node[n]['type'] for n in self.G.nodes()]

        for ntype in ntypes:
            type_dict[ntype] = self.get_nodes_of_type(ntype)
        return type_dict

    def get_neighbors(self, n, ntypes=None, return_names=False, return_ntypes=False):
        """Returns a list of neighbors of node n.
        The optional ntypes list restricts the list of neighbors to nodes that are of 
        those types.
        If return_names==True, returns the node names in addition to ids.
        If return_ntypes==True, returns the node types in addition to ids.

        Example:
        """
        if ntypes:
            neighbor_ids = [nid for nid in self.G.neighbors(n) if self.node_type(nid) in ntypes]
        else:
            neighbor_ids = self.G.neighbors(n)

        if return_names and return_ntypes:
            return [(nid, self.node_name(nid), self.node_type(nid)) for nid in neighbor_ids]
        elif return_names:
            return [(nid, self.node_name(nid)) for nid in neighbor_ids]
        elif return_ntypes:
            return [(nid, self.node_type(nid)) for nid in neighbor_ids]
        else:
            return neighbor_ids

    def get_topic(self, n):
        """Returns the topic string of an interaction.
        """
        topics = [item[1] for item in self.get_neighbors(n, ntypes=['topic'], return_names=True)]
        return topics

    def get_aux_info(self, n=None, ntypes=[], return_names=True):
        """
        """
        if n is not None:
            aux_dict = defaultdict(list)
            neighbors = self.get_neighbors(n, ntypes, return_names=True, return_ntypes=True)
            for (nid, name, ntype) in neighbors:
                if return_names:
                    aux_dict[ntype].append((nid, name))
                else:
                    aux_dict[ntype].append(nid)
            return aux_dict
        else:
            aux_all_dict = {}
            for n in self.G.nodes():
                aux_dict = defaultdict(list)
                neighbors = self.get_neighbors(n, ntypes, return_names=True, return_ntypes=True)
                for (nid, name, ntype) in neighbors:
                    if return_names:
                        aux_dict[ntype].append((nid, name))
                    else:
                        aux_dict[ntype].append(nid)
                aux_all_dict[n] = aux_dict
            return aux_all_dict

    def find_all_entity_attribute_pairs(self, subtypes=[], return_names=False, return_chids=False):
        """Find all pairs of entity -- attribute connections for given subtypes.
            If subtypes is empty, then return all.
        """

        pairs = []
        for n1, n2 in self.G.edges():
            if self.node_type(n1, 'entity') and self.node_type(n2, 'attribute'):
                if subtypes and self.G.node[n2]['subtype'] not in subtypes:
                    continue

                if return_chids:
                    pairs.append(((self.G.node[n1]['chid'], n1), n2))
                elif return_names:
                    pairs.append((self.node_name(n1), self.node_name(n2)))
                else:
                    pairs.append((n1, n2))

        return pairs

    def get_characters(self, reverse=False, only_entities=False, include_node_ids=False):
        """Get a list of all characters in the clip. Returns a list of [(name, chid)].
            If reverse==True, returns [(chid, name)]
            If only_entities==True, returns a list of character with node-ids
        """

        characters, entities = [], []
        for n in self.G.nodes():
            if self.node_type(n, 'entity'):
                entities.append(n)
                name = self.G.node[n]['name']
                chid = self.G.node[n]['chid']
                if name not in characters:
                    if reverse:
                        if include_node_ids:
                            characters.append((n, chid, name))
                        else:
                            characters.append((chid, name))
                    else:
                        if include_node_ids:
                            characters.append((n, name, chid))
                        else:
                            characters.append((name, chid))

        if only_entities:
            return entities
        return characters

    def find_all_triplets(self, int_or_rel='relationship', collapse_bidirectional=False, return_names=False):
        """Find all triplets with relationship or interaction in between.
        """

        assert(int_or_rel in ['interaction', 'relationship', 'summary'])

        triplets = []
        for n1, n2 in self.G.edges():
            # n1 --> n2 --> (n3?)
            if self.node_type(n1, 'entity') and self.node_type(n2, int_or_rel):
                for n3 in self.G.adj[n2].keys():
                    if n3 != n1 and self.node_type(n3, 'entity'):
                        if (n1, n2, n3) not in triplets:
                            if return_names:
                                triplets.append((self.node_name(n1), self.node_name(n2), self.node_name(n3)))
                            else:
                                triplets.append((n1, n2, n3))

        if collapse_bidirectional:
            # collapse bi-directional relationships
            for t in triplets:
                if (t[2], t[1], t[0]) in triplets:
                    triplets.pop(triplets.index((t[2], t[1], t[0])))

        return triplets

    def check_graph_contains_attribute(self, ch_node, st, val=None):
        """Check if an attribute attached to character already exists in the graph. Used for static propagation.
            ch_node: node number to which attached
            st: attribute subtype
            val: attribute value
            NOTE tries to find either the same subtype:value or any subtype:___
        """

        # check if node (or same subtype) exists and is connected to character
        for n in self.G.nodes():
            if self.node_type(n, 'attribute') and \
                (self.node_name(n, st + ':' + val) or self.G.node[n]['subtype'] == st) and \
                (ch_node, n) in self.G.edges():
                return True

        return False

    def check_graph_contains_relationship(self, ch_node_pair, val=None):
        """Check if a relationship attached to characters already exists in the graph. Used for static propagation.
            ch_node_pair: [n1, n2] check for a relationship n1 --> r --> n2
            val: value of expected relationship
        """

        # modify relationship triplets to {(n1, n2):r}
        rel_triplets = self.find_all_triplets(int_or_rel='relationship', collapse_bidirectional=False)
        rel_triplets = {(n1, n2):r for n1, r, n2 in rel_triplets}

        # if there is some relationship between considered node pair
        if (ch_node_pair[0], ch_node_pair[1]) in rel_triplets.keys():
            return True
        else:
            return False

    def new_nodeid(self):
        """Get an unused nodeid for new node.
        """

        return max(self.G.nodes()) + 1

    def new_nodepos(self, new_nodeid, att_conn=None, rel_conn=None):
        """Get position for a new node.
            att_conn: the character node n to which new attribute is connected
            rel_conn: the character nodes (n1, n2) to which new relationship is connected
        """

        if not att_conn == None:
            # within a circle around the character node with radius 25 -- 100
            radius = 50 + np.random.rand(1)[0]*50
            angle = np.random.rand(1)[0] * 2* np.pi
            return self.G.node[att_conn]['pos'] + np.array([radius*np.cos(angle), radius*np.sin(angle)])

        elif not rel_conn == None:
            # within a box framed by coordinates of the two character points
            # random numbers between 0.1, 0.9 makes sure they are not too close
            n1pos = self.G.node[rel_conn[0]]['pos']
            n2pos = self.G.node[rel_conn[1]]['pos']
            r = 0.1 + (np.random.rand(2)*0.8)
            return [r[0]*n1pos[0] + (1-r[0])*n2pos[0], r[1]*n1pos[1] + (1-r[1])*n2pos[1]]

        else:
            return [10, 10]

    def resolve_edges(self, rel_directions=None):
        """Resolve different kinds of missing information in edges.
            1. Make all attributes bi-directional.
            2. When r1 == r2 and there exist a -> r1 -> b and b -> r2 -> a, collapse to a <-> r1 <-> b.
            3. Collapse transitivity. a <-> r <-> b, b <-> r <-> c, c <-> r <-> a, should all use only one r node.
            4. Make certain relationships bi-directional.
        """

        # 1. Make all attributes bi-directional.
        for n1, n2 in self.G.edges():
            if self.node_type(n1, 'entity') and self.node_type(n2, 'attribute'):
                self.G.add_edge(n2, n1)
            if self.node_type(n2, 'entity') and self.node_type(n1, 'attribute') :
                self.G.add_edge(n1, n2)

        # 2. If r1 == r2 and there exist a -> r1 -> b and b -> r2 -> a, collapse to a <-> r1 <-> b.
        def check_triplets_collapse(t_type):
            triplets = self.find_all_triplets(t_type)
            for t1, t2 in itertools.combinations(triplets, 2):
                #TODO: bug, t1[2] should be checked with t2[0]
                if t1[1] != t2[1] and t1[0] == t2[2] and t1[2] == t1[0] and \
                   self.G.node[t1[1]]['name'] == self.G.node[t2[1]]['name']:
                    # print(t1, t2, "check")
                    raise RuntimeError('Unnecessary extra ' + t_type + ' node.')
                    # TODO: Waiting for example to collapse this.

        check_triplets_collapse('relationship')
        check_triplets_collapse('interaction')

        # 3. Collapse transitivity. a <-> r <-> b, b <-> r <-> c, c <-> r <-> a, should all use only one r node.
        # TODO: This is also quite complex to find automatically for more than 3 nodes. Let it be for now.
        # TODO: Find an instance of this happening, and then I'll try and fix it :)

        # 4. Make certain relationships bi-directional.
        if rel_directions:
            rel_triplets = self.find_all_triplets('relationship')
            for e1, rel, e2 in rel_triplets:
                # if name not in list, continue
                if self.G.node[rel]['name'] not in rel_directions.keys():
                    continue

                # if directed triplet, check (e2, rel, e1) doesn't exist
                if rel_directions[self.G.node[rel]['name']] == 'directed':
                    if (e2, rel, e1) in rel_triplets:
                        raise RuntimeError('should be directed, was undirected')

                # if undirected triplet, check (e2, rel, e1) exists
                if self.G.node[rel]['name'] in rel_directions.keys() and rel_directions[self.G.node[rel]['name']] == 'undirected':
                    if (e2, rel, e1) not in rel_triplets:
                        self.G.add_edge(e2, rel)
                        self.G.add_edge(rel, e1)

    def fix_spelling(self, spell_checker):
        """Fixup all the different labels for the clip using the spell-checker.
            NOTE: The spell-checker automatically makes everything lower case.
                  VOCAB does not contain other things.
        """

        def fix_label(tag):
            words = re.findall(r'\w+', tag.lower())
            fixed = [spell_checker.correction(w) for w in words]
            if words != fixed:
                print("ORI:", words)
                print("FIX:", fixed)
            return ' '.join(fixed)

        # situation
        self.situation = fix_label(self.situation)

        # scene
        self.scene_label = fix_label(self.scene_label)

        # attributes
        for n in self.G.nodes():
            if self.node_type(n, 'attribute'):
                self.G.node[n]['name'] = fix_label(self.G.node[n]['name'])

        # interactions
        for n in self.G.nodes():
            if self.node_type(n, 'interaction'):
                self.G.node[n]['name'] = fix_label(self.G.node[n]['name'])

        # relationships
        for n in self.G.nodes():
            if self.node_type(n, 'relationship'):
                self.G.node[n]['name'] = fix_label(self.G.node[n]['name'])

        # TODO: characters, maybe ignore?
        # TODO: description??

    def lemmatize(self):
        """Performs in-place lemmatization of all components of the graph.
        """

        self.situation = lemmatizer.lemmatize_situation(self.situation)
        self.scene_label = lemmatizer.lemmatize_scene(self.scene_label)

        # Each of these lemmatization functions modifies the graph G in place.
        # They can change the names of nodes, as well as delete nodes.
        lemmatizer.lemmatize_all_interactions(self.G)
        lemmatizer.lemmatize_all_attributes(self.G)
        lemmatizer.lemmatize_all_relationships(self.G)

    # def visualize_graph(self, identifier=None, prop_labels=False):
    #    """Generate networkx visualization for the graph.
    #        identifier: sid (printed on top right corner)
    #        prop_labels: visualization with highlight of nodes that were propagated
    #    """

    #    node_colors = {'entity': '#008800', 'attribute': '#880000', 'relationship': '#000088',
    #                   'interaction': '#D26A06', 'reason': '#5A143C', 'summary': '#FF00FB',
    #                   'action': '#000000', 'topic': '#FFDF0D', 'time': '#9F9F9F',
    #                   'new_attribute': '#EEFF00', 'new_relationship': '#00FFAA',
    #                   'scene': '#808080', 'situation': '#808080'}

    #    # get information to plot slowly, colors, labels, etc.
    #    colors, node_positions, labels, label_positions = [], {}, {}, {}
    #    for n, v in self.G.node.iteritems():
    #        if 'propagated' in v.keys() and v['propagated']:
    #            colors.append(node_colors['new_' + v['type']])
    #        else:
    #            colors.append(node_colors[v['type']])
    #        node_positions[n] = v['pos']
    #        labels[n] = v['origtext']
    #        label_positions[n] = (v['pos'][0], v['pos'][1]+20)

    #    # do the plotting
    #    fig = plt.gcf()
    #    nx.draw_networkx_nodes(self.G, node_positions, node_size=100, node_color=colors)
    #    nx.draw_networkx_edges(self.G, node_positions)
    #    if not prop_labels:
    #        nx.draw_networkx_labels(self.G, label_positions, labels, font_size=9, font_family='serif')
    #    else:  # draw node_labels only for "propagated" and "entity" nodes
    #        new_nodes = []
    #        for n in self.G.nodes():
    #            if self.node_type(n, 'entity') or ('propagated' in self.G.node[n].keys() and self.G.node[n]['propagated']):
    #                new_nodes.append(n)
    #        new_labels = {n:v for n, v in labels.iteritems() if n in new_nodes}
    #        new_label_positions = {n:v for n, v in label_positions.iteritems() if n in new_nodes}
    #        nx.draw_networkx_labels(self.G, new_label_positions, new_labels, font_size=9, font_family='serif')

    #    # add text labels
    #    if identifier:
    #        plt.text(fig.gca().get_xlim()[1], fig.gca().get_ylim()[1], identifier, color='#880000')
    #    plt.text(fig.gca().get_xlim()[0], fig.gca().get_ylim()[0], self.scene_label, color='#008800')
    #    plt.text(fig.gca().get_xlim()[0], fig.gca().get_ylim()[0]-25, self.situation, color='#000088')
    #    plt.axis('off')

    def pprint(self):
        """Pretty-print clip-graph.
        """

        print("Clips:", self.video['fname'])
        print("Situation:", self.situation)
        print("Scene label:", self.scene_label)
        print("Description:", self.description[:80])
        print("Graph information:")
        print("  Characters:", sum([1 for n in self.G.nodes() if self.node_type(n, 'entity')]))
        print("  Relationships:", sum([1 for n in self.G.nodes() if self.node_type(n, 'relationship')]))
        print("  Interactions:", sum([1 for n in self.G.nodes() if self.node_type(n, 'interaction')]))
        print("  Attributes:", sum([1 for n in self.G.nodes() if self.node_type(n, 'attribute')]))
        print("  Actions:", sum([1 for n in self.G.nodes() if self.node_type(n, 'action')]))


class MovieGraph(object):
    """A collection of multiple ClipGraphs for the full movie.
    """

    def __init__(self, imdb_key, castlist=None):
        """Initialize a movie graph.
        """

        self.imdb_key = imdb_key
        if castlist:
            self.castlist = castlist
        self.clip_graphs = OrderedDict()

    def attach_information(self, castlist=None, mergers=None, scenes_gt=None, sid_clip=None):
        """Attach the specified information.
        """

        if castlist:    self.castlist = castlist
        if mergers:     self.mergers = mergers
        if scenes_gt:   self.scenes_gt = scenes_gt
        if sid_clip:    self.sid_clip = sid_clip

    def add_clip_graph(self, idx, clip_graph):
        """Add a ClipGraph indexed by idx.
        """
        self.clip_graphs.update({idx: clip_graph})

    def cleanup_NA(self, verbose=False):
        """Load the scenes.gt file, and remove ClipGraphs marked N/A.
            NOTE: Make sure to attach scenes_gt information prior to calling this function.
        """

        ### Get rid of empty clip-graphs.
        # An empty clip graph has no situation label, and has 0 nodes.
        count = len(self.clip_graphs)
        for sid, cg in self.clip_graphs.iteritems():
            if not cg.situation and len(cg.G.nodes()) == 0:
                self.clip_graphs.pop(sid)
        if verbose:
            print("Popping empty clip-graphs: {} --> {}".format(count, len(self.clip_graphs)))

        ### Get rid of things marked N/A
        count = len(self.clip_graphs)
        for sid, info in self.scenes_gt.iteritems():
            if not info['use'] and sid in self.clip_graphs.keys():
                # if there are less than 3 nodes, or no situation label, ignore the N/A clip
                if len(self.clip_graphs[sid].G.nodes()) < 3 \
                   or not self.clip_graphs[sid].situation:
                    self.clip_graphs.pop(sid)
        if verbose:
            print("Popping N/A and mostly empty clip-graphs: %d --> %d".format(count, len(self.clip_graphs)))

    def perform_mergers(self, verbose=False):
        """Merge clip graphs.
            TODO: When multiple graphs present, currently merges labels to largest graph.
                  Ignores graph annotations in all other clips!
        """

        count = len(self.clip_graphs)
        merge_stats = {'zero': 0, 'one': 0, 'more': 0}
        for m_sid in self.mergers:
            sid_in_cg = [s in self.clip_graphs.keys() for s in m_sid]
            # if neither sid are in clip-graphs, continue
            if sum(sid_in_cg) == 0:
                merge_stats['zero'] += 1

            # if one sid in cg, update video information to include all clips
            elif sum(sid_in_cg) == 1:
                merge_stats['one'] += 1
                # get sid to be retained (basically sid is in clip_graphs)
                use_sid = [s for c, s in zip(sid_in_cg, m_sid) if c][0]
                fnames = [self.sid_clip[s] for s in m_sid]
                self.clip_graphs[use_sid].update_video_information(fnames)

            # if more than one sid in cg, combine smartly
            elif sum(sid_in_cg) > 1:
                merge_stats['more'] += 1
                # get list of sids with a clip-graph, find biggest graph and merge to this
                exist_sid = [s for c, s in zip(sid_in_cg, m_sid) if c]
                graph_sizes = [len(self.clip_graphs[s].G.nodes()) for s in exist_sid]
                use_sid = exist_sid[graph_sizes.index(max(graph_sizes))]

                # gather situation, scene, descriptions over all exist clips
                situ, scen, desc = [], [], []
                for s in exist_sid:
                    situ.append(self.clip_graphs[s].situation)
                    scen.append(self.clip_graphs[s].scene_label)
                    desc.append(self.clip_graphs[s].description)
                    # pop graphs which are not to be retained
                    if s != use_sid:
                        self.clip_graphs.pop(s)

                # keep a unique list of labels for situ, scenes; join for descriptions.
                self.clip_graphs[use_sid].situation = '; '.join(set(situ))
                self.clip_graphs[use_sid].scene_label = '; '.join(set(scen))
                self.clip_graphs[use_sid].description = ' '.join(desc)

                # update video info
                fnames = [self.sid_clip[s] for s in m_sid]
                self.clip_graphs[use_sid].update_video_information(fnames)

        if verbose:
            print("#Mergers:")
            print("  No graph:", merge_stats['zero'])
            print("  One graph:", merge_stats['one'])
            print("  More than one graph:", merge_stats['more'])
            print("Merging clip-graphs: {} --> {}".format(count, len(self.clip_graphs)))

    def lemmatize_all_clips(self):
        """Lemmatize all the clip graphs in the movie graph.
        """
        for sid, cg in self.clip_graphs.iteritems():
            try:
                cg.lemmatize()
            except ValueError as e:
                print("ValueError @ {}:{}".format(self.imdb_key, sid))
                print(e)

    def mine_static_information(self):
        """Attributes (age/gender/ethn/app) might be static. Reload on update if any.
        Relationships (parent/colleague) might be static. Reload on updates (time:end?).
            scene-id : character : attribute
            scene-id : relationship : character 1 --> character 2
                (bi-directional relationships will appear twice c1 --> c2, c2 --> c1)
        """

        #TODO: keep and use information about the tim:start, tim:end attributes on relationships

        self.static_info = {'rel': [], 'att': []}
        att_subtypes = ['age', 'gen', 'eth', 'pro']

        for sid, cg in self.clip_graphs.iteritems():
            ### collect attribute associations
            att_pairs = cg.find_all_entity_attribute_pairs(subtypes=att_subtypes)
            save_att = []
            for p in att_pairs:
                # save attribute
                save_att.append((sid, (p[0], cg.G.node[p[0]]['chid'],    cg.G.node[p[0]]['name']),
                                      (p[1], cg.G.node[p[1]]['subtype'], cg.G.node[p[1]]['name'])))
            # sid, (n, chid, character), (n, sub-type, attribute-name)
            self.static_info['att'].extend(save_att)

            ### collect relationship triplets
            rel_triplets = cg.find_all_triplets(int_or_rel='relationship', collapse_bidirectional=False)
            save_rel = []
            for t in rel_triplets:
                # sid, (r, relationship), (n1, chid2, character1), (n2, chid2, character2)
                # save relationship
                save_rel.append((sid, (t[1], cg.G.node[t[1]]['name']),
                                 (t[0], cg.G.node[t[0]]['chid'], cg.G.node[t[0]]['name']),
                                 (t[2], cg.G.node[t[2]]['chid'], cg.G.node[t[2]]['name'])))

            self.static_info['rel'].extend(save_rel)

    def index_static_info_by_character(self):
        """Initial static information is indexed by time (sid).
        Converting this to character-based indexing helps in propagation.
        """

        static_characters = {'rel': {}, 'att': {}}

        ### relationships --- (ch1, ch2): {time, reln}
        for rel in self.static_info['rel']:
            key = (rel[2][1:], rel[3][1:])
            val = (rel[0], rel[1][1])
            if key in static_characters['rel'].keys():
                static_characters['rel'][key].append(val)
            else:
                static_characters['rel'][key] = [val]

        ### attributes --- ch1: {time, att}
        for att in self.static_info['att']:
            key = att[1][1:]
            val = (att[0], att[2][1:])
            if key in static_characters['att'].keys():
                static_characters['att'][key].append(val)
            else:
                static_characters['att'][key] = [val]

        self.static_characters = static_characters

    def propagate_static_labels(self, verbose=False):
        """Propagate static labels that are mined to all other ClipGraphs.
        """

        new_nodes = {'att': 0, 'rel': 0}

        for sid, cg in self.clip_graphs.iteritems():
            if verbose:
                print("================== {} ==================".format(sid))

            # get list of characters
            clip_ch = cg.get_characters(reverse=True)
            ch_nodes = cg.get_characters(only_entities=True)

            ### Forward relationships for all pairs of characters
            for pair in itertools.permutations(clip_ch, 2):
                # no relationships for this pair? continue
                if pair not in self.static_characters['rel'].keys():
                    continue

                # get ordered character nodes for pair
                ch_node_pair = [n for p in pair for n in ch_nodes if cg.node_name(n, p[1])]

                # go through reversed (time-wise) list of relationships for that pair, and add when missing
                for rel in reversed(self.static_characters['rel'][pair]):
                    t, val = rel
                    if t >= sid:
                        continue  # ignore relationships mined after current/future clip
                    else:
                        # check if graph contains the same relationship or a different one
                        in_graph = cg.check_graph_contains_relationship(ch_node_pair, val)
                        if not in_graph:
                            # create a new relationship node
                            new_nodeid = cg.new_nodeid()
                            cg.G.add_node(new_nodeid, name=val, origtext=val)
                            cg.G.add_node(new_nodeid, type='relationship', propagated=True)
                            cg.G.add_node(new_nodeid, pos=cg.new_nodepos(new_nodeid, rel_conn=ch_node_pair))
                            # add edges
                            cg.G.add_edge(ch_node_pair[0], new_nodeid)
                            cg.G.add_edge(new_nodeid, ch_node_pair[1])
                            # check if bidirectional
                            rpair = (pair[1], pair[0])  # get reversed pair to index
                            if rpair in self.static_characters['rel'].keys():
                                for rrel in self.static_characters['rel'][rpair]:  # check every relationship pair
                                    # if same time and value, then bidirectional!
                                    if t == rrel[0] and val == rrel[1]:
                                        cg.G.add_edge(ch_node_pair[1], new_nodeid)
                                        cg.G.add_edge(new_nodeid, ch_node_pair[0])
                            # count
                            if verbose:
                                print("+REL (attached to {}-rel-{}):".format(ch_node_pair[0], ch_node_pair[1]), cg.G.node[new_nodeid])
                            new_nodes['rel'] += 1

            ### Forward attributes for each character in clip
            for ch in clip_ch:
                # no static attributes stored for this character? continue!
                if ch not in self.static_characters['att'].keys():
                    continue

                # get character nodeid
                ch_node = [n for n in ch_nodes if cg.node_name(n, ch[1])][0]

                # go through reversed (time-wise) list of attributes, and add missing
                for att in reversed(self.static_characters['att'][ch]):
                    t, (st, val) = att
                    if t >= sid:
                        continue  # ignore attributes mined from current/future clips
                    else:
                        # check if graph contains the same attribute, or
                        # something with same sub-type that overrides proposed addition
                        in_graph = cg.check_graph_contains_attribute(ch_node, st, val)
                        if not in_graph:
                            # create a new attribute node
                            new_nodeid = cg.new_nodeid()
                            cg.G.add_node(new_nodeid, name=val, type='attribute', propagated=True)
                            cg.G.add_node(new_nodeid, subtype=st, origtext=st + ':' + val)
                            cg.G.add_node(new_nodeid, pos=cg.new_nodepos(new_nodeid, att_conn=ch_node))
                            # add edges
                            cg.G.add_edge(new_nodeid, ch_node)
                            cg.G.add_edge(ch_node, new_nodeid)
                            # count
                            if verbose:
                                print("+ATT (attached to {}):".format(ch_node), cg.G.node[new_nodeid])
                            new_nodes['att'] += 1

        print("SUMMARY: Added {} attribute nodes, {} relationship nodes.".format(new_nodes['att'], new_nodes['rel']))

    # def visualize_all_graphs(self, fname='test.pdf', prop_labels=False):
    #    """Generate a pdf output file consisting of all graphs.
    #    """

    #    with PdfPages(fname, 'w') as write_pdf:
    #        for sid, clip_graph in self.clip_graphs.iteritems():
    #            try:
    #                plt.figure(figsize=(8,5))
    #                clip_graph.visualize_graph(str(sid), prop_labels=prop_labels)
    #                write_pdf.savefig()
    #                plt.close()
    #            except:
    #                print("cannot plot for", str(sid))

    def count_occurrences(self):
        """Returns lists of all types of nodes.
        """

        # get all character-labels
        char_labels = []
        for k, cg in self.clip_graphs.iteritems():
            char_labels.extend([cg.G.node[n]['name'] for n in cg.G.nodes() if cg.node_type(n, 'entity')])

        # get all situation-labels
        situ_labels = []
        for k, cg in self.clip_graphs.iteritems():
            situ_labels.append(cg.situation)

        # get all relationship-labels
        reln_labels = []
        for k, cg in self.clip_graphs.iteritems():
            reln_labels.extend([cg.G.node[n]['name'] for n in cg.G.nodes() if cg.node_type(n, 'relationship')])

        # get all interaction-labels
        intr_labels = []
        for k, cg in self.clip_graphs.iteritems():
            intr_labels.extend([cg.G.node[n]['name'] for n in cg.G.nodes() if cg.node_type(n, 'interaction')])

        # get all summary-labels
        summary_labels = []
        for k, cg in self.clip_graphs.iteritems():
            summary_labels.extend([cg.G.node[n]['name'] for n in cg.G.nodes() if cg.node_type(n, 'summary')])

        # get all attribute-labels
        attr_labels = {'emo': [], 'app': [], 'age': [], 'gen': [], 'eth': [], 'pro': []}
        for k, cg in self.clip_graphs.iteritems():
            for at in attr_labels.keys():
                attr_labels[at].extend([cg.G.node[n]['name'] for n in cg.G.nodes() if cg.node_type(n, 'attribute') and cg.G.node[n]['subtype'] in [at]])

        # collect and return
        collect = {'situations': situ_labels,
                   'characters': char_labels,
                   'interactions': intr_labels,
                   'relationships': reln_labels,
                   'summaries': summary_labels}
        for att_type, values in attr_labels.iteritems():
            collect.update({'attributes:' + att_type: values})

        return collect

