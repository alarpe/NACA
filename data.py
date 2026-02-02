# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pickle
import json
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Tuple, List, Dict
from os import PathLike

import torch.nn as nn
import torch.nn.functional as F 
import math,copy
from os import PathLike
from typing import Sequence, Dict, Union, Tuple, List
import collections.abc as container_abcs
from torch_geometric.data import Data, Batch, Dataset
from scipy.spatial.qhull import Delaunay
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import utils
from torchvision.utils import make_grid

import matplotlib.tri as tri

class MeshAirfoilDataset(Dataset):
    def __init__(self, root_path: str, mode: str = 'train'):
        """Initializes dataset with mesh graph data and files."""
        super().__init__(root_path)
        self.mode = mode
        self.data_dir = Path(root_path) / ('outputs_' + mode)
        self.file_list = os.listdir(self.data_dir)

        print(len(self.file_list))
        files = self.file_list.copy()
        for element in files:
            if ".pkl" not in element:
                self.file_list.remove(element)
        print("size of data: ",len(self.file_list))

        self.len = len(self.file_list)
        with open(self.data_dir.parent / 'train_max_min.pkl', 'rb') as f:
            self.normalization_factors = pickle.load(f)

        #
        factor_file = self.data_dir.parent / 'factors.pkl'
        if not os.path.exists(factor_file):
            self.a = None
            self.b = None
        else:
            with open(factor_file,'rb') as filename:
                factors = pickle.load(filename)
                self.a,self.b = factors

    def __len__(self):
        return self.len

    def get(self, idx: int, mode=1) -> Data:
        """Returns a data object representing the mesh graph with field information."""

        length_scale = 1.0 #
        with open(self.data_dir / self.file_list[idx], 'rb') as f:
            field_data = pickle.load(f)
        nodes = torch.from_numpy(field_data['pos'][:,:2])/length_scale
        edges = torch.from_numpy(np.transpose(field_data['edges']))
        elems_list = field_data['elems']
        fields = []
        for i in range(field_data['fields'].shape[1]):
            fields.append(field_data['fields'][:,i])
        fields = self.preprocess(fields)

        spd,aoa = self.get_params_from_name(self.file_list[idx])
        aoa = torch.from_numpy(aoa)
        spd = torch.from_numpy(spd)
        norm_aoa = aoa / 10.0   #
        norm_spd = spd / 1.0    #
        nodes_x = torch.cat([
            nodes,
            norm_aoa.unsqueeze(0).repeat(nodes.shape[0], 1),
            norm_spd.unsqueeze(0).repeat(nodes.shape[0], 1)
        ], dim=-1)  #

        data = Data(x=nodes_x, y=fields, edge_index=edges, pos=nodes)
        self.add_edge_attributes(data)
        data.aoa = aoa
        data.norm_aoa = norm_aoa
        data.spd = spd
        data.norm_spd = norm_spd
        if mode==2:
            data.elems_list = elems_list
        return data

    def preprocess(self, tensor_list, stack_output=True):
        data_max, data_min = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            normalized = (tensor_list[i] - data_min[i]) / (data_max[i] - data_min[i]) * 2 - 1
            if isinstance(normalized, np.ndarray):
                normalized = torch.from_numpy(normalized)
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = torch.stack(normalized_tensors, dim=1)
            #print(f"normalized_tensors shape: {normalized_tensors.shape}")
        return normalized_tensors

    # get flow parameters from file
    @staticmethod
    def get_params_from_name(filename: str) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray]:
        """Parses flow conditions from filename."""
        s = filename.rsplit('.', 1)[0].split('_')
        spd = np.array([float(s[1][1:])], dtype=np.float32)
        aoa = np.array([float(s[0][3:])], dtype=np.float32)
        return spd,aoa

    def add_edge_attributes(self, data: Data):
        """Adds edge attributes to the data object for graph representation."""
        sender = data.x[data.edge_index[0]]
        receiver = data.x[data.edge_index[1]]
        relation_pos = sender[:, 0:2] - receiver[:, 0:2]
        data.edge_attr = torch.norm(relation_pos, p=2, dim=1, keepdim=True)

        if self.a==None or self.b==None: ####
            std_epsilon = torch.tensor([1e-8])
            self.a = torch.mean(data.edge_attr, axis=0, dtype=torch.float32)
            self.b = data.edge_attr.std(dim=0)
            self.b = torch.maximum(self.b, std_epsilon)
            #
            factors = np.array([self.a,self.b])
            factor_file = self.data_dir.parent / 'factors.pkl'
            with open(factor_file,'wb') as f:
                pickle.dump(factors,f)
        data.edge_attr = (data.edge_attr - self.a) / self.b

def log_images(nodes, pred, true, batch, elems_list, mode, log_idx=0, iterate=0, file='field.png', zoom=True,
                   xb=[-0.5, 1.5], yb=[-1.0, 1.0]):

    inds = batch == log_idx
    nodes = nodes[inds]
    pred = pred[inds]
    true = true[inds]
    for field in range(pred.shape[1]):
        true_img = plot_field(nodes, elems_list, true[:, field],
                            title='true', zoom=zoom, xb=xb, yb=yb)  ###
        true_img = ToTensor()(true_img)
        min_max = (true[:, field].min().item(), true[:, field].max().item())

        pred_img = plot_field(nodes, elems_list, pred[:, field],
                            title='pred', clim=min_max, zoom=zoom, xb=xb, yb=yb)  ###
        pred_img = ToTensor()(pred_img)

        #
        dif = pred[:, field] - true[:, field]
        dif_img = plot_field(nodes, elems_list, dif,
                            title='diff', zoom=zoom, xb=xb, yb=yb)  ###
        dif_img = ToTensor()(dif_img)

        imgs = [pred_img, true_img, dif_img]
        grid = make_grid(torch.stack(imgs), padding=0)
        out_file = file + f'{field}'
        utils.save_image(grid, out_file + '_field.png')

def log_images_1(nodes, pred, true, batch, elems_list, mode, log_idx=0, iterate=0, file='field.png', zoom=True,
                   xb=[-0.5, 1.5], yb=[-1.0, 1.0]):

    inds = batch == log_idx
    nodes = nodes[inds]
    pred = pred[inds]
    true = true[inds]
    for field in range(pred.shape[1]):
        min_max_t = (true[:, field].min().item(), true[:, field].max().item())
        min_max_p = (pred[:, field].min().item(), pred[:, field].max().item())
        min_max = (min(min_max_t[0],min_max_p[0]), max(min_max_t[1],min_max_p[1]))

        true_img = plot_field(nodes, elems_list, true[:, field],
                            title='true', clim=min_max, zoom=zoom, xb=xb, yb=yb)
        true_img = ToTensor()(true_img)

        pred_img = plot_field(nodes, elems_list, pred[:, field],
                            title='pred', clim=min_max, zoom=zoom, xb=xb, yb=yb)
        pred_img = ToTensor()(pred_img)

        #
        dif = pred[:, field] - true[:, field]
        min_max_d = (dif.min().item(), dif.max().item())
        dif_img = plot_field(nodes, elems_list, dif,
                            title='diff', clim=min_max_d, zoom=zoom, xb=xb, yb=yb)
        dif_img = ToTensor()(dif_img)

        imgs = [pred_img, true_img, dif_img]
        grid = make_grid(torch.stack(imgs), padding=0)
        out_file = file + f'{field}'
        utils.save_image(grid, out_file + '_field.png')

def plot_field(nodes, elems_list, field, contour=False, clim=None, zoom=True,
               get_array=True, out_file=None, show=False, title='',
               xb=[-0.5, 1.5], yb=[-1.0, 1.0]):
    #elems_list = sum(elems_list, [])
    tris, _ = quad2tri(elems_list)
    tris = np.array(tris)
    x, y = nodes[:, :2].t().detach().cpu().numpy()
    field = field.detach().cpu().numpy()
    fig = plt.figure(dpi=800)
    if contour:
        plt.tricontourf(x, y, tris, field)
    else:
        plt.tripcolor(x, y, tris, field)
    if clim:
        plt.clim(*clim)
    plt.colorbar()
    if zoom:
        plt.xlim(left=xb[0], right=xb[1])  ###
        plt.ylim(bottom=yb[0], top=yb[1])  ###
    if title:
        plt.title(title)

    if out_file is not None:
        plt.savefig(out_file)
        plt.close()

    if show:
        plt.show()
    # raise NotImplementedError

    if get_array:
        fig.canvas.draw()
        a = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
        a = a.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return a

def quad2tri(elems):
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e)==3:
        #if len(e) <= 3:
            new_elems.append(e)
        elif len(e)>3:
        #else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(torch.tensor(([[e[0]], [e[2]]]), dtype=torch.long))
    new_edges = torch.cat(new_edges, dim=1) if new_edges else torch.tensor([], dtype=torch.long)
    return new_elems, new_edges
