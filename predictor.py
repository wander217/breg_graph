import copy
import json
import random
import torch
import numpy as np
import dgl
import yaml
from typing import Dict, List
from loss_model import LossModel
from dataset import GraphAlphabet, GraphLabel, norm
from torch import Tensor
import warnings
import argparse
import time


class BREGPredictor:
    def __init__(self, config: str, alphabet: str, label: str, pretrained: str):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            data: Dict = yaml.safe_load(f)
        self.alphabet: GraphAlphabet = GraphAlphabet(alphabet_path=alphabet)
        self.label: GraphLabel = GraphLabel(label_path=label)
        self.model = LossModel(vocab=self.alphabet.size(),
                               class_num=self.label.size(),
                               device=self.device,
                               **data['loss_model'])
        state_dict = torch.load(pretrained, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

    def _process_ocr(self, ocr_result: Dict):
        texts = []
        lengths = []
        bboxes = []
        for item in ocr_result['target']:
            encoded_label: np.ndarray = self.alphabet.encode(item["text"])
            texts.append(encoded_label)
            lengths.append(encoded_label.shape[0])
            bbox = np.array(item["bbox"]).astype(np.int32).flatten().tolist()
            x = bbox[0::2]
            w = np.max(x) - np.min(x)
            y = bbox[1::2]
            h = np.max(y) - np.min(y)
            bbox = np.concatenate([bbox, [h, w]], axis=-1)
            bboxes.append(bbox)
        return np.array(texts), np.array(lengths), np.array(bboxes)

    def _preprocess(self, ocr_result: Dict):
        texts, lengths, boxes = self._process_ocr(ocr_result)
        node_nums = texts.shape[0]
        src = []
        dst = []
        edges = []
        for i in range(node_nums):
            for j in range(node_nums):
                if i == j:
                    continue
                y_dist = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                x_dist = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                h = boxes[i, 8]
                if np.abs(y_dist) > 3 * h:
                    continue
                edges.append([x_dist, y_dist])
                src.append(i)
                dst.append(j)
        # print("-"*55)
        # for a, b in zip(src, dst):
        #     print("+"*55)
        #     print(a, ocr_result["target"][a]["label"], ocr_result["target"][a]["text"])
        #     print(b, ocr_result["target"][b]["label"], ocr_result["target"][b]["text"])
        #     print("+" * 55)
        # print("-" * 55)
        edges = np.array(edges)
        graphs = dgl.DGLGraph()
        graphs.add_nodes(node_nums)
        graphs.add_edges(src, dst)

        boxes = norm(boxes)
        edges = norm(edges)
        boxes = torch.from_numpy(boxes).float()
        edges = torch.from_numpy(edges).float()
        graphs.edata['feat'] = edges
        graphs.ndata['feat'] = boxes

        node_nums = graphs.number_of_nodes()
        node_factor = torch.FloatTensor(node_nums, 1).fill_(1. / float(node_nums))
        node_factor = node_factor.sqrt()

        edge_nums = graphs.number_of_edges()
        edge_factor = torch.FloatTensor(edge_nums, 1).fill_(1. / float(edge_nums))
        edge_factor = edge_factor.sqrt()

        max_length = np.max(lengths)
        new_text = [np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), 'constant'), axis=0)
                    for t in texts]
        texts = np.concatenate(new_text)

        texts = torch.from_numpy(np.array(texts)).long()
        lengths = torch.from_numpy(np.array(lengths)).long()

        return (graphs,
                texts,
                lengths,
                node_factor,
                edge_factor,
                node_nums,
                edge_nums)

    def predict(self, ocr_result: Dict):
        input_data = self._preprocess(ocr_result)
        score: Tensor = self.model.predict(*input_data)
        values, pred = score.cpu().softmax(1).max(1)
        result = [(self.label.decode(pred[i].item()), values[i].item())
                  for i in range(len(pred))]
        pred_ocr = copy.deepcopy(ocr_result)
        for ocr, label in zip(pred_ocr["target"], result):
            ocr["label"] = label[0]
            ocr["label_score"] = label[1]
        return pred_ocr


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser("Predictor config")
    parser.add_argument("-c", "--config_path", type=str, default='', help="config path")
    parser.add_argument("-r", "--resume", type=str, default='', help="checkpoint path")
    parser.add_argument("-a", "--alphabet_path", type=str, default='', help="alphabet path")
    parser.add_argument("-l", "--label_path", type=str, default='', help="label path")
    parser.add_argument("-i", "--input", type=str, default='', help="input data path")
    args = parser.parse_args()
    predictor = BREGPredictor(args.config_path.strip(),
                              args.alphabet_path.strip(),
                              args.label_path.strip(),
                              args.resume.strip())
    with open(args.input.strip(), 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
    start = time.time()
    id = random.randint(0, len(data) - 1)
    output = predictor.predict(data[id])
    print(data[id]['file_name'])
    print("Run_time:", time.time() - start)
    wrong: int = 0
    for pred, gt in zip(output['target'], data[id]['target']):
        print("-" * 50)
        print("Pred:", pred["text"], pred['label'], pred['label_score'])
        print("GT:", gt["text"], gt['label'])
        if pred['label'] != gt['label']:
            wrong += 1
        print("-" * 50)
    print("Wrong", wrong)
    print("Total", len(data[id]['target']))
