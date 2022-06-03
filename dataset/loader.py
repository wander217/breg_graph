import dgl
import torch
import json
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from dataset.alphabet import GraphAlphabet, GraphLabel
from utils import remove_space


def norm(element: np.ndarray) -> np.ndarray:
    """
    :param element: bounding box coordinate: (n, 8) or (n, 2)
    :return: normed element
    """
    mn = np.min(element, axis=0)
    mx = np.max(element, axis=0)
    normed_point = (element - mn) / (mx - mn)
    normed_element = (normed_point - 0.5) / 0.5
    return normed_element


def process(sample: Dict,
            label_dict: GraphLabel,
            alphabet_dict: GraphAlphabet):
    """
    :param sample: a dict containing keys:
        - img: image path
        - target: A list containing bbox information.
                  Each bbox is containing:
                  +, label:  label of this bbox
                  +, text: text inside this bbox
                  +, bbox: a polygon having type (8, 2)
    :param label_dict: use to encode label
    :param alphabet_dict: use to encode text
    :return:
        - bboxes: contain 4 point coordinate including width and height
        - labels: contain label of bounding box in batch
        - texts: contain text of batch
        - lengths: contain length of each text in batch
    """
    TARGET_KEY = "target"
    TEXT_KEY = "text"
    LABEL_KEY = "label"
    BBOX_KEY = "bbox"
    SHAPE_KEY = "shape"

    lengths = []
    texts = []
    bboxes = []
    labels = []
    for target in sample[TARGET_KEY]:
        text = alphabet_dict.encode(target[TEXT_KEY])
        texts.append(text)
        lengths.append(text.shape[0])
        # assert text.shape[0] > 0, text.shape[0]
        label: int = label_dict.encode(target[LABEL_KEY])
        labels.append(label)
        bbox = np.array(target[BBOX_KEY]).astype(np.int32).flatten().tolist()
        x = bbox[0::2]
        w = np.max(x) - np.min(x)
        y = bbox[1::2]
        h = np.max(y) - np.min(y)
        bbox = np.concatenate([bbox, [w, h]], axis=-1).flatten()
        bboxes.append(bbox)
    return (np.array(bboxes),
            np.array(labels),
            np.array(texts),
            np.array(lengths))


class GraphDataset(Dataset):
    def __init__(self,
                 path: str,
                 alphabet: GraphAlphabet,
                 label: GraphLabel):
        self._ldict: GraphLabel = label
        self._adict: GraphAlphabet = alphabet
        self._graphs: List = []
        self._texts: List = []
        self._lengths: List = []
        self._labels: List = []
        self._load(path)

    def _load(self, target_path: str):
        with open(target_path, 'r', encoding='utf-8') as f:
            samples: List = json.loads(remove_space(f.readline()))

        for sample in samples:
            bboxes, labels, texts, lengths = process(sample, self._ldict, self._adict)
            node_size = labels.shape[0]
            src: List = []
            dst: List = []
            dists: List = []
            for i in range(node_size):
                x_i = np.mean(bboxes[i][:8][0::2])
                y_i = np.mean(bboxes[i][:8][1::2])
                w_i = bboxes[i][9]
                h_i = bboxes[i][8]
                for j in range(node_size):
                    if i == j:
                        continue

                    x_j = np.mean(bboxes[j][:8][0::2])
                    y_j = np.mean(bboxes[j][:8][1::2])
                    h_j = bboxes[j][9]
                    w_j = bboxes[j][8]
                    x_dist = x_j - x_i
                    y_dist = y_j - y_i

                    if np.abs(y_dist) > 3 * h_j:
                        continue
                    dists.append([x_dist, y_dist, lengths[j] / lengths[i]])
                    src.append(i)
                    dst.append(j)
            g = dgl.DGLGraph()
            g.add_nodes(node_size)
            g.add_edges(src, dst)
            g.ndata['feat'] = torch.FloatTensor(norm(bboxes))
            g.edata['feat'] = torch.FloatTensor(norm(np.array(dists)))

            self._graphs.append(g)
            self._texts.append(texts)
            self._lengths.append(lengths)
            self._labels.append(labels)

    def __getitem__(self, index: int):
        graph: dgl.DGLGraph = self._graphs[index]
        text: np.ndarray = self._texts[index]
        length: List = self._lengths[index]
        label: int = self._labels[index]
        return graph, label, text, length

    def __len__(self):
        return len(self._graphs)


def get_factor(sizes: List) -> Tensor:
    tab_snorm: List = [torch.ones((size, 1)).float() / float(size)
                       for size in sizes]
    factor: Tensor = torch.cat(tab_snorm).sqrt()
    return factor


def graph_collate(batch: Tuple, pad_encode: int = 0):
    graphs, labels, texts, lengths = map(list, zip(*batch))
    labels = np.concatenate(labels)
    lengths = np.concatenate(lengths)
    max_len: int = np.max(lengths)
    texts = np.concatenate(texts)
    new_text: List = [np.expand_dims(np.pad(text,
                                            (0, max_len - text.shape[0]),
                                            'constant',
                                            constant_values=pad_encode), axis=0)
                      for text in texts]
    texts = np.concatenate(new_text)
    node_sizes = [graph.number_of_nodes() for graph in graphs]
    node_factor: Tensor = get_factor(node_sizes)
    edge_sizes = [graph.number_of_edges() for graph in graphs]
    edge_factor: Tensor = get_factor(edge_sizes)
    batched_graph = dgl.batch(graphs)

    print("okie lắm bạn ơi")
    return (batched_graph,
            torch.from_numpy(labels),
            torch.from_numpy(texts),
            torch.from_numpy(lengths),
            node_factor,
            edge_factor,
            node_sizes,
            edge_sizes)


class GraphLoader:
    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 drop_last: bool,
                 shuffle: bool,
                 pin_memory: bool,
                 dataset: Dict,
                 alphabet: GraphAlphabet,
                 label: GraphLabel):
        self.dataset: GraphDataset = GraphDataset(**dataset,
                                                  alphabet=alphabet,
                                                  label=label)
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=graph_collate
        )
