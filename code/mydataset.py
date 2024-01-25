import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.knowledge_graph import KnowledgeGraphDataset, _read_dictionary, _read_triplets, _read_triplets_as_list, build_knowledge_graph
import numpy as np

from dgl.data.utils import (
    _get_dgl_url,
    deprecate_function,
    deprecate_property,
    download,
    extract_archive,
    generate_mask_tensor,
    get_download_dir,
    load_graphs,
    load_info,
    makedirs,
    save_graphs,
    save_info,
)

import pandas as pd
import logging
logger = logging.getLogger(__name__)

















def _read_triplets_as_labels(filename):
    l = []
    for triplet in _read_triplets(filename):
        
        score = float(triplet[3])
        l.append([score])
    return l

def _read_triplets_as_diff_labels(filename):
    l = []
    for triplet in _read_triplets(filename):
        
        score = float(triplet[4])
        l.append([score])
    return l


class JDDataset(KnowledgeGraphDataset):
    """KnowledgeGraph link prediction dataset

    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18

    Parameters
    -----------
    name : str
        Name can be 'FB15k-237', 'FB15k' or 'wn18'.
    reverse : bool
        Whether add reverse edges. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self,
        name,
        reverse=True,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
        train_path=None,
        eval_path=None,
        test_path=None,
    ):
        self._name = name
        self.reverse = reverse
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
        super(JDDataset, self).__init__(
            name,
            reverse=reverse,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
    def download(self):
        pass

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
        By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return self.raw_dir

    def process(self):
        """
        The original knowledge base is stored in triplets.
        This function will parse these triplets and build the DGLGraph.
        """
        root_path = self.raw_path
        entity_path = os.path.join(root_path, "entities.dict")
        relation_path = os.path.join(root_path, "relations.dict")
        skill2cluster_path = os.path.join(root_path, "skill2cluster.tsv")
        train_path = os.path.join(self.train_path)
        valid_path = os.path.join(self.eval_path)
        test_path = os.path.join(self.test_path)
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        skill2cluster = pd.read_csv(skill2cluster_path, header=None)[0]
        train = np.asarray(
            _read_triplets_as_list(train_path, entity_dict, relation_dict)
        )
        valid = np.asarray(
            _read_triplets_as_list(valid_path, entity_dict, relation_dict)
        )
        test = np.asarray(
            _read_triplets_as_list(test_path, entity_dict, relation_dict)
        )
        edge_labels = torch.tensor(_read_triplets_as_labels(train_path) + _read_triplets_as_labels(valid_path) + _read_triplets_as_labels(test_path)).squeeze(-1)
        if 'diff' in train_path:
            diff_labels = torch.tensor(_read_triplets_as_diff_labels(train_path) + _read_triplets_as_diff_labels(valid_path) + _read_triplets_as_diff_labels(test_path)).squeeze(-1)
        num_nodes = len(entity_dict)
        num_rels = len(relation_dict)
        if self.verbose:
            logger.info("
            logger.info("
            logger.info("
            logger.info("
            logger.info("

        
        self._train = train
        self._valid = valid
        self._test = test

        self._num_nodes = num_nodes
        self._num_rels = num_rels
        
        g, data = build_knowledge_graph(
            num_nodes, num_rels, train, valid, test, reverse=self.reverse
        )
        (
            etype,
            ntype,
            train_edge_mask,
            valid_edge_mask,
            test_edge_mask,
            train_mask,
            val_mask,
            test_mask,
        ) = data
        g.edata["train_edge_mask"] = train_edge_mask
        g.edata["valid_edge_mask"] = valid_edge_mask
        g.edata["test_edge_mask"] = test_edge_mask
        g.edata["train_mask"] = train_mask
        g.edata["eval_mask"] = val_mask
        g.edata["test_mask"] = test_mask
        g.edata["etype"] = etype
        g.edata['edge_labels'] = edge_labels
        if 'diff' in train_path:
            g.edata['diff_labels'] = diff_labels.long()
        g.ndata["ntype"] = ntype
        g.ndata['nodes'] = torch.tensor([int(i) for i in entity_dict])
        g.ndata['ncluster'] = torch.tensor([-1]* (num_nodes - len(skill2cluster)) + list(skill2cluster))

        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True

        return False

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        save_graphs(str(graph_path), self._g)
        save_info(
            str(info_path),
            {"num_nodes": self.num_nodes, "num_rels": self.num_rels},
        )

    def load(self):
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        self._num_nodes = info["num_nodes"]
        self._num_rels = info["num_rels"]
        self._g = graphs[0]
        train_mask = self._g.edata["train_edge_mask"].numpy()
        val_mask = self._g.edata["valid_edge_mask"].numpy()
        test_mask = self._g.edata["test_edge_mask"].numpy()

        
        self._g.edata["train_edge_mask"] = generate_mask_tensor(
            self._g.edata["train_edge_mask"].numpy()
        )
        self._g.edata["valid_edge_mask"] = generate_mask_tensor(
            self._g.edata["valid_edge_mask"].numpy()
        )
        self._g.edata["test_edge_mask"] = generate_mask_tensor(
            self._g.edata["test_edge_mask"].numpy()
        )
        self._g.edata["train_mask"] = generate_mask_tensor(
            self._g.edata["train_mask"].numpy()
        )
        self._g.edata["val_mask"] = generate_mask_tensor(
            self._g.edata["val_mask"].numpy()
        )
        self._g.edata["test_mask"] = generate_mask_tensor(
            self._g.edata["test_mask"].numpy()
        )

        
        etype = self._g.edata["etype"].numpy()
        self._etype = etype
        u, v = self._g.all_edges(form="uv")
        u = u.numpy()
        v = v.numpy()
        train_idx = np.nonzero(train_mask == 1)
        self._train = np.column_stack(
            (u[train_idx], etype[train_idx], v[train_idx])
        )
        valid_idx = np.nonzero(val_mask == 1)
        self._valid = np.column_stack(
            (u[valid_idx], etype[valid_idx], v[valid_idx])
        )
        test_idx = np.nonzero(test_mask == 1)
        self._test = np.column_stack(
            (u[test_idx], etype[test_idx], v[test_idx])
        )

        if self.verbose:
            print("
            print("
            print("
            print("
            print("

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def save_name(self):
        return self.name + "_dgl_graph"
    
    def save(self):
        r"""Overwite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``dgl.data.utils.save_graphs``
        to save dgl graph into files and use
        ``dgl.data.utils.save_info`` to save extra
        information into files.
        """
        pass




class JD_task3_Dataset(KnowledgeGraphDataset):
    """KnowledgeGraph link prediction dataset

    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18

    Parameters
    -----------
    name : str
        Name can be 'FB15k-237', 'FB15k' or 'wn18'.
    reverse : bool
        Whether add reverse edges. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self,
        name,
        reverse=True,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
        train_path=None,
        eval_path=None,
        test_path=None,
        idx=None
    ):
        self._name = name
        self.reverse = reverse
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
        self.idx = idx
        super(JD_task3_Dataset, self).__init__(
            name,
            reverse=reverse,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
    def download(self):
        pass

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
        By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return self.raw_dir

    def process(self):
        """
        The original knowledge base is stored in triplets.
        This function will parse these triplets and build the DGLGraph.
        """
        root_path = self.raw_path
        entity_path = os.path.join(root_path, "entities.dict")
        relation_path = os.path.join(root_path, "relations.dict")
        idx = self.idx
        
        paths = [os.path.join(root_path, f"{i}/triplet_percentage.tsv") for i in range(1, 6)]
        eval_path = os.path.join(root_path, "6/triplet_percentage.tsv")
        test_path = os.path.join(root_path, "7/triplet_percentage.tsv")

        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        
        trains = [np.asarray(_read_triplets_as_list(paths[i], entity_dict, relation_dict) for i in range(1, 6))]
        valid = np.asarray(
            _read_triplets_as_list(eval_path, entity_dict, relation_dict)
        )
        test = np.asarray(
            _read_triplets_as_list(test_path, entity_dict, relation_dict)
        )
        edge_labels = torch.tensor(_read_triplets_as_labels(paths[0]) + _read_triplets_as_labels(paths[1]) + _read_triplets_as_labels(paths[2]) +_read_triplets_as_labels(paths[3]) +_read_triplets_as_labels(paths[4]) +_read_triplets_as_labels(paths[5]) + _read_triplets_as_labels(paths[6])).squeeze(-1)
        num_nodes = len(entity_dict)
        num_rels = len(relation_dict)
        train = [j for i in trains for j in i]
        valid = [j for i in trains for j in i]
        test = [j for i in trains for j in i]
        if self.verbose:
            logger.info("
            logger.info("
            logger.info("
            logger.info("
            logger.info("

        
        self._train = train
        self._valid = valid
        self._test = test

        self._num_nodes = num_nodes
        self._num_rels = num_rels
        
        g, data = build_knowledge_graph(
            num_nodes, num_rels, train, valid, test, reverse=self.reverse
        )
        (
            etype,
            ntype,
            train_edge_mask,
            valid_edge_mask,
            test_edge_mask,
            train_mask,
            val_mask,
            test_mask,
        ) = data
        g.edata["train1_edge_mask"] = train_edge_mask
        g.edata["train2_edge_mask"] = train_edge_mask
        g.edata["train3_edge_mask"] = train_edge_mask
        g.edata["train4_edge_mask"] = train_edge_mask
        g.edata["train5_edge_mask"] = train_edge_mask
        g.edata["train6_edge_mask"] = train_edge_mask
        g.edata["test_edge_mask"] = test_edge_mask
        g.edata["train_mask"] = train_mask
        g.edata["eval_mask"] = val_mask
        g.edata["test_mask"] = test_mask
        g.edata["etype"] = etype
        g.edata['edge_labels'] = edge_labels
        g.ndata["ntype"] = ntype
        g.ndata['nodes'] = torch.tensor([int(i) for i in entity_dict])
        

        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True

        return False

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        save_graphs(str(graph_path), self._g)
        save_info(
            str(info_path),
            {"num_nodes": self.num_nodes, "num_rels": self.num_rels},
        )

    def load(self):
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        self._num_nodes = info["num_nodes"]
        self._num_rels = info["num_rels"]
        self._g = graphs[0]
        train_mask = self._g.edata["train_edge_mask"].numpy()
        val_mask = self._g.edata["valid_edge_mask"].numpy()
        test_mask = self._g.edata["test_edge_mask"].numpy()

        
        self._g.edata["train_edge_mask"] = generate_mask_tensor(
            self._g.edata["train_edge_mask"].numpy()
        )
        self._g.edata["valid_edge_mask"] = generate_mask_tensor(
            self._g.edata["valid_edge_mask"].numpy()
        )
        self._g.edata["test_edge_mask"] = generate_mask_tensor(
            self._g.edata["test_edge_mask"].numpy()
        )
        self._g.edata["train_mask"] = generate_mask_tensor(
            self._g.edata["train_mask"].numpy()
        )
        self._g.edata["val_mask"] = generate_mask_tensor(
            self._g.edata["val_mask"].numpy()
        )
        self._g.edata["test_mask"] = generate_mask_tensor(
            self._g.edata["test_mask"].numpy()
        )

        
        etype = self._g.edata["etype"].numpy()
        self._etype = etype
        u, v = self._g.all_edges(form="uv")
        u = u.numpy()
        v = v.numpy()
        train_idx = np.nonzero(train_mask == 1)
        self._train = np.column_stack(
            (u[train_idx], etype[train_idx], v[train_idx])
        )
        valid_idx = np.nonzero(val_mask == 1)
        self._valid = np.column_stack(
            (u[valid_idx], etype[valid_idx], v[valid_idx])
        )
        test_idx = np.nonzero(test_mask == 1)
        self._test = np.column_stack(
            (u[test_idx], etype[test_idx], v[test_idx])
        )

        if self.verbose:
            print("
            print("
            print("
            print("
            print("

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def save_name(self):
        return self.name + "_dgl_graph"
    
    def save(self):
        r"""Overwite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``dgl.data.utils.save_graphs``
        to save dgl graph into files and use
        ``dgl.data.utils.save_info`` to save extra
        information into files.
        """
        pass
