from dataclasses import dataclass, field
from typing import List
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.nn import EGConv

@dataclass
class DataConfig:
    seq_dir: str = './RNAdesignv1/train/seqs/'
    npy_data_dir: str = './RNAdesignv1/train/coords/'

@dataclass
class ModelConfig:
    smoothing: float = 0.1
    hidden: int = 128
    vocab_size: int = 4
    k_neighbors: int = 30   # 暂时不动
    dropout: float = 0.1    # 暂时不动
    node_feat_types: List[str] = field(default_factory=lambda: ['angle', 'distance', 'direction','contact'])
    edge_feat_types: List[str] = field(default_factory=lambda: ['orientation', 'distance', 'direction'])
    num_encoder_layers: int = 5
    num_decoder_layers: int = 5

@dataclass
class TrainConfig:
    batch_size: int = 8
    epoch: int = 150
    lr: float = 0.001
    output_dir: str = './model_v4'
    ckpt_path: str = './model_v4/best.pt'

@dataclass
class Config:
    pipeline: str = 'train'
    device: str = 'cuda:7'
    data_config: DataConfig = field(default_factory=DataConfig)       
    model_config: ModelConfig = field(default_factory=ModelConfig)    
    train_config: TrainConfig = field(default_factory=TrainConfig) 

####################data.py#####################
def read_fasta_biopython(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def process_data(config):
    train_file_list = os.listdir(config.data_config.seq_dir)
    content_dict = {"pdb_id": [], "seq": []}
    for file in tqdm(train_file_list):
        sequences = read_fasta_biopython(config.data_config.seq_dir + file)
        content_dict["pdb_id"].append(list(sequences.keys())[0])
        content_dict["seq"].append(list(sequences.values())[0])

    data = pd.DataFrame(content_dict)
    split = np.random.choice(['train', 'valid', 'test'], size=len(data), p=[0.9, 0.1, 0])
    data['split'] = split
    train_data = data[data['split'] == 'train']
    valid_data = data[data['split'] == 'valid']
    test_data = data[data['split'] == 'test']
    return train_data, valid_data, test_data

class RNADataset(Dataset):
    def __init__(self, data, npy_dir):
        super(RNADataset, self).__init__()
        self.data = data
        self.npy_dir = npy_dir
        self.seq_list = self.data['seq'].to_list()
        self.name_list = self.data['pdb_id'].to_list()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        pdb_id = self.name_list[idx]
        coords = np.load(os.path.join(self.npy_dir, pdb_id + '.npy'))
        feature = {
            "name": pdb_id,
            "seq": seq,
            "coords": {
                "P": coords[:, 0, :],
                "O5'": coords[:, 1, :],
                "C5'": coords[:, 2, :],
                "C4'": coords[:, 3, :],
                "C3'": coords[:, 4, :],
                "O3'": coords[:, 5, :],
            }
        }
        return feature

def featurize(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    names = []

    for i, b in enumerate(batch):
        x = np.stack([np.nan_to_num(b['coords'][c], nan=0.0) for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        X[i, :, :, :] = x_pad
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        names.append(b['name'])

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X) + np.nan
    for i, n in enumerate(numbers):
        X_new[i, :n, :] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask, lengths, names

def get_dataloaders(config):
    # 数据处理
    config=Config()
    train_data, valid_data, test_data=process_data(config)
    data_config = config.data_config
    train_config = config.train_config

    train_dataset = RNADataset(data=train_data, npy_dir=data_config.npy_data_dir)
    valid_dataset = RNADataset(data=valid_data, npy_dir=data_config.npy_data_dir)
    test_dataset = RNADataset(data=test_data, npy_dir=data_config.npy_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=0, collate_fn=featurize)
    valid_loader = DataLoader(valid_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0, collate_fn=featurize)
    test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0, collate_fn=featurize)

    return train_loader, valid_loader, test_loader

##########################模型#######################
class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.ReLU()

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id=None):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_E)))))
        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))
        return h_V

def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features



class RNAFeatures(nn.Module):
    def __init__(self, edge_features, node_features, node_feat_types=[], edge_feat_types=[], num_rbf=16, top_k=30, augment_eps=0., dropout=0.1, args=None):
        super(RNAFeatures, self).__init__()
        """Extract RNA Features"""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.dropout = nn.Dropout(dropout)
        self.node_feat_types = node_feat_types
        self.edge_feat_types = edge_feat_types
        self.feat_dims = {
                'node': {
                    'angle': 12,
                    'distance': 80,
                    'direction': 9,
                    'contact': 16
                },
                'edge': {
                    'orientation': 4,
                    'distance': 96,
                    'direction': 15,
                }
            }
        node_in = sum([self.feat_dims['node'][feat] for feat in node_feat_types])
        edge_in = sum([self.feat_dims['edge'][feat] for feat in edge_feat_types])
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)
        # 接触图编码模块（参考网页10的GVP架构）
        self.contact_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.feat_dims['node']['contact'])
        )
        
    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx
    
    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    
    def _get_rbf(self, A, B, E_idx=None, num_rbf=16):
        if E_idx is not None:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
            D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0]
            RBF_A_B = self._rbf(D_A_B_neighbors)
        else:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,:,None,:])**2,-1) + 1e-6)
            RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B
    
    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q
    
    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        V = X.clone()
        X = X[:,:,:6,:].reshape(X.shape[0], 6*X.shape[1], 3) 
        dX = X[:,1:,:] - X[:,:-1,:]
        U = _normalize(dX, dim=-1)
        u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
        n_0 = _normalize(torch.cross(u_0, u_1, dim=-1), dim=-1)
        b_1 = _normalize(u_0 - u_1, dim=-1)
        
        # select C3'
        n_0 = n_0[:,4::6,:] 
        b_1 = b_1[:,4::6,:]
        X = X[:,4::6,:]

        Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0, dim=-1)), 2)
        Q = Q.view(list(Q.shape[:2]) + [9])
        Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [16, 464, 9]

        Q_neighbors = gather_nodes(Q, E_idx) # [16, 464, 30, 9]
        P_neighbors = gather_nodes(V[:,:,0,:], E_idx) # [16, 464, 30, 3]
        O5_neighbors = gather_nodes(V[:,:,1,:], E_idx)
        C5_neighbors = gather_nodes(V[:,:,2,:], E_idx)
        C4_neighbors = gather_nodes(V[:,:,3,:], E_idx)
        O3_neighbors = gather_nodes(V[:,:,5,:], E_idx)
        
        Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [16, 464, 1, 3, 3]
        Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3,3]) # [16, 464, 30, 3, 3]

        dX = torch.stack([P_neighbors,O5_neighbors,C5_neighbors,C4_neighbors,O3_neighbors], dim=3) - X[:,:,None,None,:] # [16, 464, 30, 3]
        dU = torch.matmul(Q[:,:,:,None,:,:], dX[...,None]).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
        B, N, K = dU.shape[:3]
        E_direct = _normalize(dU, dim=-1)
        E_direct = E_direct.reshape(B, N, K,-1)
        R = torch.matmul(Q.transpose(-1,-2), Q_neighbors)
        E_orient = self._quaternions(R)
        
        dX_inner = V[:,:,[0,2,3],:] - X.unsqueeze(-2)
        dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
        dU_inner = _normalize(dU_inner, dim=-1)
        V_direct = dU_inner.reshape(B,N,-1)
        return V_direct, E_direct, E_orient
    
    def _dihedrals(self, X, eps=1e-7):
        # P, O5', C5', C4', C3', O3'
        X = X[:,:,:6,:].reshape(X.shape[0], 6*X.shape[1], 3)
        dX = X[:, 5:, :] - X[:, :-5, :] # O3'-P, P-O5', O5'-C5', C5'-C4', ...
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]  # O3'-P, P-O5', ...
        u_1 = U[:,1:-1,:] # P-O5', O5'-C5', ...
        u_0 = U[:,2:,:]   # O5'-C5', C5'-C4', ...
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        
        D = F.pad(D, (3,4), 'constant', 0)
        D = D.view((D.size(0), D.size(1) //6, 6))
        return torch.cat((torch.cos(D), torch.sin(D)), 2) # return D_features
    
    def _compute_contact_map(self,X, cutoff=8.0):
        """
        X: 输入的三维坐标张量 [B, L, 6, 3]，包含P、O5'、C5'等原子坐标
        cutoff: 接触判定阈值（单位Å），RNA常用5-10Å[6,7](@ref)
        """
        if X.dim() == 3:
            X = X.unsqueeze(0)  # 添加批次维度
        B, L = X.shape[0], X.shape[1]
        
        # 提取C4'原子坐标（网页11的基准点）
        c4_coords = X[:, :, 3, :]  # 维度: [B, L, 3]
        
        # 批量计算距离矩阵（优化内存）
        dist_matrix = torch.cdist(c4_coords, c4_coords)  # [B, L, L]
        
        # 生成接触图（8Å阈值[11](@ref)）
        contact_map = (dist_matrix <= cutoff).float()
        contact_map = contact_map * (1 - torch.eye(L, device=X.device).unsqueeze(0))
        
        return contact_map

    def forward(self, X, S, mask):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)   # 高斯噪音

        # 动态维度处理
        assert X.size(-2) == 6, "输入必须包含6个原子坐标(P,O5',C5',C4',C3',O3')"
        assert mask.shape == X.shape[:2], "Mask维度不匹配输入数据"
        B, N = X.shape[0], X.shape[1]
        # Build k-Nearest Neighbors graph
        B, N, _,_ = X.shape
        # P, O5', C5', C4', C3', O3'
        atom_P = X[:, :, 0, :]
        atom_O5_ = X[:, :, 1, :]
        atom_C5_ = X[:, :, 2, :]
        atom_C4_ = X[:, :, 3, :]
        atom_C3_ = X[:, :, 4, :] 
        atom_O3_ = X[:, :, 5, :]

        X_backbone = atom_P
        
        D_neighbors, E_idx = self._dist(X_backbone, mask)        

        mask_bool = (mask==1)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        contact_map = self._compute_contact_map(X)  # [B, L, L]
        contact_feat = contact_map.sum(dim=-1, keepdim=True)  # [B, L, 1]
        contact_feat = node_mask_select(self.contact_proj(contact_feat))  # [B, L, 16]

        # node features
        h_V = []
        # angle
        V_angle = node_mask_select(self._dihedrals(X))
        # distance
        node_list = ['O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        V_dist = []
        
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            V_dist.append(node_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(V_dist), dim=-1).squeeze()
        # direction
        V_direct, E_direct, E_orient = self._orientations_coarse(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct, E_orient = list(map(lambda x: edge_mask_select(x), [E_direct, E_orient]))

        # edge features
        h_E = []
        # dist
        edge_list = ['P-P', 'O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        E_dist = [] 
        for pair in edge_list:
            atom1, atom2 = pair.split('-')
            E_dist.append(edge_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)))
        E_dist = torch.cat(tuple(E_dist), dim=-1)

        if 'angle' in self.node_feat_types:
            h_V.append(V_angle)
        if 'distance' in self.node_feat_types:
            h_V.append(V_dist)
        if 'direction' in self.node_feat_types:
            h_V.append(V_direct)
        if 'contact' in self.node_feat_types:
            h_V.append(contact_feat)

        if 'orientation' in self.edge_feat_types:
            h_E.append(E_orient)
        if 'distance' in self.edge_feat_types:
            h_E.append(E_dist)
        if 'direction' in self.edge_feat_types:
            h_E.append(E_direct)
            
        # Embed the nodes
        h_V = self.norm_nodes(self.node_embedding(torch.cat(h_V, dim=-1)))
        h_E = self.norm_edges(self.edge_embedding(torch.cat(h_E, dim=-1)))

        # prepare the variables to return
        S = torch.masked_select(S, mask_bool)
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]
        return X, S, h_V, h_E, E_idx, batch_id

class RNAModel(nn.Module):
    def __init__(self, model_config):
        super(RNAModel, self).__init__()
        self.smoothing = model_config.smoothing
        self.node_features = self.edge_features = model_config.hidden
        self.hidden_dim = model_config.hidden
        self.vocab = model_config.vocab_size

        self.features = RNAFeatures(
            model_config.hidden, model_config.hidden,
            top_k=model_config.k_neighbors,
            dropout=model_config.dropout,
            node_feat_types=model_config.node_feat_types,
            edge_feat_types=model_config.edge_feat_types,
            args=model_config
        )

        self.W_s = nn.Embedding(model_config.vocab_size, self.hidden_dim)
        self.encoder_layers = nn.ModuleList([
            MPNNLayer(self.hidden_dim, self.hidden_dim * 2, dropout=model_config.dropout)
            for _ in range(model_config.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            MPNNLayer(self.hidden_dim, self.hidden_dim * 2, dropout=model_config.dropout)
            for _ in range(model_config.num_decoder_layers)])

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )

        self.readout = nn.Linear(self.hidden_dim, model_config.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask)
        for enc_layer in self.encoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc_layer(h_V, h_EV, E_idx, batch_id)

        for dec_layer in self.decoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec_layer(h_V, h_EV, E_idx, batch_id)

        graph_embs = []
        for b_id in range(batch_id[-1].item() + 1):
            b_data = h_V[batch_id == b_id].mean(0)
            graph_embs.append(b_data)
        graph_embs = torch.stack(graph_embs, dim=0)
        graph_prjs = self.projection_head(graph_embs)

        logits = self.readout(h_V)
        return logits, S, graph_prjs

    def sample(self, X, S, mask=None):
        X, gt_S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask)
        for enc_layer in self.encoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc_layer(h_V, h_EV, E_idx, batch_id)

        for dec_layer in self.decoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec_layer(h_V, h_EV, E_idx, batch_id)

        logits = self.readout(h_V)
        return logits, gt_S
    
##########################训练######################
def train(config, train_loader, valid_loader):
    model = RNAModel(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), config.train_config.lr)
    criterion = nn.CrossEntropyLoss()
    train_config = config.train_config

    if not os.path.exists(train_config.output_dir):
        os.makedirs(train_config.output_dir)

    best_valid_recovery = 0
    for epoch in range(train_config.epoch):
        model.train()
        epoch_loss = 0
        train_pbar = train_loader
        for batch in train_pbar:
            X, S, mask, lengths, names = batch
            X = X.to(config.device)
            S = S.to(config.device)
            mask = mask.to(config.device)
            logits, S, _ = model(X, S, mask)
            loss = criterion(logits, S)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        model.eval()
        with torch.no_grad():
            recovery_list = []
            for batch in valid_loader:
                X, S, mask, lengths, names = batch
                X = X.to(config.device)
                S = S.to(config.device)
                mask = mask.to(config.device)
                logits, S, _ = model(X, S, mask)
                probs = F.softmax(logits, dim=-1)
                samples = probs.argmax(dim=-1)
                start_idx = 0
                for length in lengths:
                    end_idx = start_idx + length.item()
                    sample = samples[start_idx:end_idx]
                    gt_S = S[start_idx:end_idx]
                    recovery = (sample == gt_S).sum() / len(sample)
                    recovery_list.append(recovery.cpu().numpy())
                    start_idx = end_idx
            valid_recovery = np.mean(recovery_list)
            print(f'Epoch {epoch + 1}/{train_config.epoch}|Loss: {epoch_loss:.4f}|recovery: {valid_recovery:.4f}')
            if valid_recovery > best_valid_recovery:
                best_valid_recovery = valid_recovery
                torch.save(model.state_dict(), os.path.join(train_config.output_dir, 'best.pt'))

def test(config, test_loader):
    eval_model = RNAModel(config.model_config).to(config.device)
    checkpoint_path = config.train_config.ckpt_path
    print(f"Loading checkpoint from path: {checkpoint_path}")
    eval_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    eval_model.to(config.device)
    eval_model.eval()

    with torch.no_grad():
        recovery_list = []
        for batch in test_loader:
            X, S, mask, lengths, names = batch
            X = X.to(config.device)
            S = S.to(config.device)
            mask = mask.to(config.device)
            logits, S, _ = eval_model(X, S, mask)
            probs = F.softmax(logits, dim=-1)
            samples = probs.argmax(dim=-1)
            start_idx = 0
            for length in lengths:
                end_idx = start_idx + length.item()
                sample = samples[start_idx:end_idx]
                gt_S = S[start_idx:end_idx]
                recovery = (sample == gt_S).sum() / len(sample)
                recovery_list.append(recovery.cpu().numpy())
                start_idx = end_idx
        test_recovery = np.mean(recovery_list)
        print(f'Test recovery: {test_recovery:.4f}')

def main():
    config = Config()

    # 获取数据加载器
    train_loader, valid_loader, test_loader = get_dataloaders(config)

    if config.pipeline == 'train':
        train(config, train_loader, valid_loader)
    elif config.pipeline == 'test':
        test(config, test_loader)
    else:
        raise ValueError(f"Unsupported pipeline: {config.pipeline}")

if __name__ == "__main__":
    main()