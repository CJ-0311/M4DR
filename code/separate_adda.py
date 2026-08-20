import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import math
import warnings
import sys
import torch.nn.functional as F
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)
module_path = "~/jupyterlab/scdeal"
if module_path not in sys.path:
    sys.path.append(module_path)
# 確保您的 loss.py 和 mmd.py 位於Python路徑中
# 假設 loss.py 和 mmd.py 位於同一目錄
import loss as custom_loss
import mmd as custom_mmd
import coral as custom_coral 

# =============================================================================
# 0. 配置參數 (Configuration)
# =============================================================================
class Config:
    # --- 文件路徑 ---
    BULK_EXPR_PATH = '~/jupyterlab/GSE140440_du145/GSE140440du145_hvg4000_gdsc.csv'
    DRUG_SMILES_PATH = '~/jupyterlab/alldrugname_with_smiles.csv'
    DRUG_RESPONSE_PATH = '~/jupyterlab/gdsc1.3_less.csv'
    SINGLE_CELL_EXPR_PATH = '~/jupyterlab/GSE140440_du145/GSE140440du145tpm.csv'
    BULK_PATHWAY_PATH = '~/jupyterlab/bulk_pathwayscoreMatrix.csv'
    SC_PATHWAY_PATH = '~/jupyterlab/GSE140441/GSE140440du145_pathwayscoreMatrix.csv'

    # --- 訓練超參數 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRETRAIN_EPOCHS = 100
    FINETUNE_EPOCHS = 150
    BATCH_SIZE = 32  # 由於模型變複雜，減小Batch Size以防內存不足
    LR_PRETRAIN = 3e-5  # Transformer類模型通常需要更小的學習率
    LR_FINETUNE = 1e-5
    WEIGHT_DECAY = 1e-5  # 加入權重衰減
    LAMBDA = 0.5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    OUTPUT_DIR = '/home/bio_wangxf/jupyterlab/GSE140440_du145/outputs/'
    # --- 特徵維度 ---
    FP_SIZE = 2048  # 摩根指紋維度


# =============================================================================
# 1. 損失函數 (與之前相同)
# =============================================================================
loss_dict = {"CORAL": custom_coral.CORAL}


# =============================================================================
# 2. 數據加載和預處理 (已更新)
# =============================================================================


def load_and_prepare_data(config):
    print("--- 1. Starting Data Loading and Preprocessing (Advanced) ---")
    # 加載所有數據
    df_bulk_expr = pd.read_csv(config.BULK_EXPR_PATH, index_col=0)
    df_smiles = pd.read_csv(config.DRUG_SMILES_PATH, usecols=['alldrugname', 'smiles']).dropna()
    
    # --- START OF MODIFICATION ---
    # 在讀取 CSV 時，將 'NA', 'N/A' 等常見缺失值字串直接轉換為 np.nan
    na_values = ['NA', 'N/A', 'NaN', 'nan', '']
    df_response = pd.read_csv(config.DRUG_RESPONSE_PATH, index_col=0, na_values=na_values)
    # --- END OF MODIFICATION ---



    df_sc_expr = pd.read_csv(config.SINGLE_CELL_EXPR_PATH, index_col=0)
    df_bulk_pathway = pd.read_csv(config.BULK_PATHWAY_PATH, index_col=0)
    df_sc_pathway = pd.read_csv(config.SC_PATHWAY_PATH, index_col=0)

    # 預處理 (轉置，重命名等)
    df_bulk_expr.index = df_bulk_expr.index.str.lower().str.strip()
    df_bulk_expr.columns = df_bulk_expr.columns.str.lower().str.strip()
    df_response.index = df_response.index.str.lower().str.strip()
    df_response.columns = df_response.columns.str.lower().str.strip()
    df_smiles['alldrugname'] = df_smiles['alldrugname'].str.lower().str.strip()

    # --- 单独处理通路数据 ---
    # 对df_sc_pathway和df_sc_expr进行转置和标准化
    df_sc_expr = df_sc_expr.T
    df_sc_expr.index = df_sc_expr.index.str.lower().str.replace('-', '.').str.strip()
    df_sc_expr.columns = df_sc_expr.columns.str.lower().str.replace('-', '.').str.strip()

    df_sc_pathway = df_sc_pathway.T
    df_sc_pathway.index = df_sc_pathway.index.str.lower().str.replace('-', '.').str.strip()
    df_sc_pathway.columns = df_sc_pathway.columns.str.lower().str.replace('-', '.').str.strip()

    # 对df_bulk_pathway不进行转置，只进行行列名标准化
    df_bulk_pathway.index = df_bulk_pathway.index.str.lower().str.replace('-', '.').str.strip()
    df_bulk_pathway.columns = df_bulk_pathway.columns.str.lower().str.replace('-', '.').str.strip()
    # -----------------------

    # --- 數據對齊 ---
    common_drugs = sorted(list(set(df_smiles['alldrugname']) & set(df_response.columns)))
    common_bulk_cells = sorted(list(set(df_bulk_expr.index) & set(df_response.index) & set(df_bulk_pathway.index)))
    common_genes = sorted(list(set(df_bulk_expr.columns) & set(df_sc_expr.columns)))
    common_pathways = sorted(list(set(df_bulk_pathway.columns) & set(df_sc_pathway.columns)))

    print(f"Number of common drugs: {len(common_drugs)}")
    print(f"Number of common bulk cells: {len(common_bulk_cells)}")
    print(f"Number of common genes: {len(common_genes)}")
    print(f"Number of common pathways: {len(common_pathways)}")
    # 過濾數據
    df_smiles = \
    df_smiles[df_smiles['alldrugname'].isin(common_drugs)].drop_duplicates(subset=['alldrugname']).set_index(
        'alldrugname').loc[common_drugs]
    df_response = df_response.loc[common_bulk_cells, common_drugs]
    df_bulk_expr = df_bulk_expr.loc[common_bulk_cells, common_genes]
    df_sc_expr = df_sc_expr[common_genes]
    df_bulk_pathway = df_bulk_pathway.loc[common_bulk_cells, common_pathways]
    df_sc_pathway = df_sc_pathway[common_pathways]

    #df_response.to_csv('gdsc1.3process.csv')
    #print("预处理后的GDSC1.3数据已保存为 'gdsc1.3process.csv'")
    
    # --- 特徵縮放 ---
    df_sc_expr = np.log2(df_sc_expr+1)
    bulk_scaler_gene = StandardScaler().fit(df_bulk_expr)
    sc_scaler_gene   = StandardScaler().fit(df_sc_expr)

    df_bulk_expr[:] = bulk_scaler_gene.transform(df_bulk_expr)
    df_sc_expr[:]   = sc_scaler_gene.transform(df_sc_expr)


    
    # --- 通路标准化（改成和基因一致）---
    bulk_scaler_pathway = StandardScaler().fit(df_bulk_pathway)
    sc_scaler_pathway   = StandardScaler().fit(df_sc_pathway)

    df_bulk_pathway[:] = bulk_scaler_pathway.transform(df_bulk_pathway)
    df_sc_pathway[:]   = sc_scaler_pathway.transform(df_sc_pathway)

    print("Checking data for extreme values...")
    print(f"Gene expression range: {df_bulk_expr.values.min()} to {df_bulk_expr.values.max()}")
    print(f"Pathway score range: {df_bulk_pathway.values.min()} to {df_bulk_pathway.values.max()}")
    print(f"Single-cell gene expression range: {df_sc_expr.values.min()} to {df_sc_expr.values.max()}")
    print(f"Single-cell pathway score range: {df_sc_pathway.values.min()} to {df_sc_pathway.values.max()}")


    # 检查标准化后单细胞基因表达数据中表达值超过50的细胞
    high_expr_cells = df_sc_expr[(df_sc_expr > 50).any(axis=1)]
    print(f"\nNumber of single cells with expression > 50 after standardization: {len(high_expr_cells)}")

    if len(high_expr_cells) > 0:
        print("These cells and their maximum expression values:")
        for cell_name in high_expr_cells.index:
            max_val = df_sc_expr.loc[cell_name].max()
            print(f"  {cell_name}: max expression = {max_val:.4f}")
        
    # 进一步分析这些异常细胞的基因表达分布
    print("\nDetailed analysis of high-expression cells:")
    for cell_name in high_expr_cells.index:
        cell_data = df_sc_expr.loc[cell_name]
        high_genes = cell_data[cell_data > 50]
        print(f"  {cell_name}: {len(high_genes)} genes > 50, max = {high_genes.max():.4f}")
        print(f"    Top 5 highly expressed genes: {high_genes.nlargest(5).to_dict()}")

    
    # 检查是否有无限值
    if np.isinf(df_bulk_expr.values).any():
        print("WARNING: Infinite values found in bulk gene expression data")
    if np.isinf(df_bulk_pathway.values).any():
        print("WARNING: Infinite values found in bulk pathway data")
    if np.isinf(df_sc_expr.values).any():
        print("WARNING: Infinite values found in single-cell gene expression data")
    if np.isinf(df_sc_pathway.values).any():
        print("WARNING: Infinite values found in single-cell pathway data")
    
    # 检查是否有NaN值
    if np.isnan(df_bulk_expr.values).any():
        print("WARNING: NaN values found in bulk gene expression data")
    if np.isnan(df_bulk_pathway.values).any():
        print("WARNING: NaN values found in bulk pathway data")
    if np.isnan(df_sc_expr.values).any():
        print("WARNING: NaN values found in single-cell gene expression data")
    if np.isnan(df_sc_pathway.values).any():
        print("WARNING: NaN values found in single-cell pathway data")
    
    # 處理藥物反應標籤和不平衡問題
    df_response_long = df_response.stack().reset_index()
    df_response_long.columns = ['cell_line', 'drug_name', 'response']
    df_response_long.dropna(inplace=True)
    df_response_long['response'] = df_response_long['response'].map({'sensitive': 1, 'resistant': 0}).astype(int)
    groups = df_response_long.groupby('drug_name')
    balanced_dfs = [pd.concat(
        [group[group['response'] == 1].sample(n=min(group.response.value_counts()), random_state=config.RANDOM_STATE),
         group[group['response'] == 0].sample(n=min(group.response.value_counts()), random_state=config.RANDOM_STATE)])
                    for _, group in groups if len(group.response.unique()) > 1]
    df_final = pd.concat(balanced_dfs).reset_index(drop=True)

    train_df, val_df = train_test_split(df_final, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                                        stratify=df_final['drug_name'])

    print("--- Data Loading and Preprocessing Finished ---")
    return train_df, val_df, df_bulk_expr, df_sc_expr, df_bulk_pathway, df_sc_pathway, df_smiles, \
           df_bulk_expr.columns, df_bulk_pathway.columns


# =============================================================================
# 3. 特徵生成與數據集類 (已更新)
# =============================================================================
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
import torch

def smiles_to_graph(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        # 1. 原子类型 (One-hot) - 约10-20维
        atomic_num = atom.GetAtomicNum()
        # 定义常见元素列表，其余归为"其他"
        common_atoms = [6, 7, 8, 9, 16, 17, 35, 53] # C, N, O, F, S, Cl, Br, I
        atom_type_feature = [1 if atomic_num == elem else 0 for elem in common_atoms]
        atom_type_feature.append(1 if atomic_num not in common_atoms else 0) # "其他"类别

        # 2. 原子度 (One-hot) - 7维 (0,1,2,3,4,5,其他)
        degree = atom.GetDegree()
        degree_feature = [1 if degree == d else 0 for d in range(6)]
        degree_feature.append(1 if degree > 5 else 0)

        # 3. 形式电荷 (One-hot) - 5维 (-2,-1,0,+1,+2)
        formal_charge = atom.GetFormalCharge()
        charge_feature = [1 if formal_charge == c else 0 for c in [-2, -1, 0, 1, 2]]

        # 4. 杂化方式 (One-hot) - 5维
        hybridization = atom.GetHybridization()
        hybrid_feature = [
            int(hybridization == HybridizationType.SP),
            int(hybridization == HybridizationType.SP2),
            int(hybridization == HybridizationType.SP3),
            int(hybridization == HybridizationType.SP3D),
            int(hybridization == HybridizationType.SP3D2)
        ]

        # 5. 芳香性 (Binary) - 1维
        is_aromatic = int(atom.GetIsAromatic())

        # 6. 氢原子数量 (One-hot) - 5维 (0,1,2,3,其他)
        num_h = atom.GetTotalNumHs()
        num_h_feature = [1 if num_h == n else 0 for n in range(4)]
        num_h_feature.append(1 if num_h > 3 else 0)

        # 7. 是否在环中 (Binary) - 1维
        is_in_ring = int(atom.IsInRing())

        # 将所有特征拼接起来
        feature_vector = (
            atom_type_feature +
            degree_feature +
            charge_feature +
            hybrid_feature +
            [is_aromatic] +
            num_h_feature +
            [is_in_ring]
        )

        atom_features_list.append(feature_vector)

    x = torch.tensor(atom_features_list, dtype=torch.float)

    # ... (你的边索引代码保持不变) ...
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


def smiles_to_fingerprint(smiles_string, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return torch.zeros(n_bits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)


class MultiModalDataset(Dataset):
    def __init__(self, dataframe, df_expr, df_pathway, graph_map, fp_map):
        self.dataframe = dataframe.reset_index(drop=True)
        self.df_expr = df_expr
        self.df_pathway = df_pathway
        self.graph_map = graph_map
        self.fp_map = fp_map

    def __len__(self): return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        cell_line, drug_name, response = row['cell_line'], row['drug_name'], row['response']
        gene_expr = torch.FloatTensor(self.df_expr.loc[cell_line].values)
        pathway_score = torch.FloatTensor(self.df_pathway.loc[cell_line].values)
        drug_graph = self.graph_map[drug_name]
        drug_fp = self.fp_map[drug_name]
        return gene_expr, pathway_score, drug_graph, drug_fp, torch.LongTensor([response])


class SCTargetDataset(Dataset):
    def __init__(self, df_expr, df_pathway):
        self.df_expr = df_expr
        self.df_pathway = df_pathway

    def __len__(self): return len(self.df_expr)

    def __getitem__(self, idx):
        gene_expr = torch.FloatTensor(self.df_expr.iloc[idx].values)
        pathway_score = torch.FloatTensor(self.df_pathway.iloc[idx].values)
        return gene_expr, pathway_score


def custom_collate_fn(batch):
    gene_exprs, pathway_scores, drug_graphs, drug_fps, responses = zip(*batch)
    return torch.stack(gene_exprs, 0), torch.stack(pathway_scores, 0), GraphDataLoader(list(drug_graphs),
                                                                                       batch_size=len(
                                                                                           drug_graphs)), torch.stack(
        drug_fps, 0), torch.cat(responses, 0)


# 3.5. NEW: ADDA-Specific Modules
# =============================================================================
class DomainDiscriminator(nn.Module):
    """Simple Discriminator for ADDA."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1) # Output a single logit for domain classification
        )

    def forward(self, features):
        return self.net(features)



# =============================================================================
# 4. 全新模型架構 (全面升級)
# =============================================================================
class GeneTransformerEncoder(nn.Module):
    def __init__(self, gene_dim, d_model=256, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(gene_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.output_proj(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),  # BatchNorm1d -> LayerNorm
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)   # BatchNorm1d -> LayerNorm
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[1024, 512, 512, 256], dropout_rate=0.4):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]), # BatchNorm1d -> LayerNorm
            nn.Dropout(dropout_rate)
        ]
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout_rate))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dims[i + 1])) # BatchNorm1d -> LayerNorm
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        # 關鍵：batch_first=True 使得輸入維度為 (batch, seq, feature)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x, key_padding_mask=None):
        # x shape: (batch_size, max_nodes_in_batch, embed_dim)
        x_norm = self.norm1(x)
        # 關鍵：傳入 key_padding_mask 告知注意力層忽略padding部分
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class GraphTransformer(nn.Module):
    # 修改 __init__ 簽名以接受 dropout
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        # 將 dropout 傳遞給每一層
        self.layers = nn.ModuleList([GraphTransformerLayer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def to_dense_batch(self, x, batch):
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = torch.bincount(batch)
        max_nodes = num_nodes_per_graph.max().item()

        # 創建一個填充好的張量和一個mask
        x_padded = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
        # key_padding_mask: True代表是padding，需要被忽略
        key_padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device)

        # 填充數據和mask
        for i in range(batch_size):
            nodes = x[batch == i]
            num_nodes = len(nodes)
            x_padded[i, :num_nodes] = nodes
            key_padding_mask[i, :num_nodes] = False  # False代表是真實數據

        return x_padded, key_padding_mask

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.input_proj(x)

        # 1. 將稀疏的圖批次轉換為密集(padded)的序列批次和對應的padding mask
        x_padded, key_padding_mask = self.to_dense_batch(x, batch)

        # 2. 將padded序列和mask傳入Transformer層
        for layer in self.layers:
            x_padded = layer(x_padded, key_padding_mask=key_padding_mask)

        # 3. 在池化前，移除padding的影響
        # 我們只對有效的節點進行池化
        # key_padding_mask中False代表有效節點，所以用~反轉
        valid_nodes_mask = ~key_padding_mask.squeeze(-1) if key_padding_mask.dim() == 3 else ~key_padding_mask
        x = x_padded[valid_nodes_mask]

        # 4. 全局池化得到圖的表示
        x = global_mean_pool(x, batch)
        x = self.output_proj(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, nhead=8, dropout=0.1):
        super().__init__()
        
        # 2a. 新增獨立的投影層
        self.cell_proj = nn.Linear(embed_dim, embed_dim)
        self.drug_proj = nn.Linear(embed_dim, embed_dim)

        # 2b. 新增獨立的 LayerNorm 層
        self.cell_norm = nn.LayerNorm(embed_dim)
        self.drug_norm = nn.LayerNorm(embed_dim)

        # 注意力層
        self.cell_drug_attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.drug_cell_attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 預測器 MLP (將 BatchNorm1d 換成 LayerNorm 以保持一致性)
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024), nn.ReLU(), nn.LayerNorm(1024), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.LayerNorm(512), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, cell_feat, drug_feat):
        # 2c. 應用 投影 + LayerNorm + L2 標準化
        cell_feat_proj = self.cell_norm(self.cell_proj(cell_feat))
        drug_feat_proj = self.drug_norm(self.drug_proj(drug_feat))
        
        cell_feat_norm = F.normalize(cell_feat_proj, p=2, dim=1)
        drug_feat_norm = F.normalize(drug_feat_proj, p=2, dim=1)
        
        # (Batch, Dim) -> (Batch, 1, Dim) for attention
        cell_feat_unsq = cell_feat_norm.unsqueeze(1)
        drug_feat_unsq = drug_feat_norm.unsqueeze(1)

        # cell "關注" drug
        cell_enhanced, _ = self.cell_drug_attention(cell_feat_unsq, drug_feat_unsq, drug_feat_unsq)
        cell_enhanced = self.norm1(cell_enhanced.squeeze(1) + cell_feat_norm)

        # drug "關注" cell
        drug_enhanced, _ = self.drug_cell_attention(drug_feat_unsq, cell_feat_unsq, cell_feat_unsq)
        drug_enhanced = self.norm2(drug_enhanced.squeeze(1) + drug_feat_norm)
        
        final_feat = torch.cat([cell_enhanced, drug_enhanced], dim=1)
        return self.output_mlp(final_feat)


class FourTowerModel(nn.Module):
    def __init__(self, gene_dim, pathway_dim, fp_dim, graph_node_dim):
        super().__init__()

        # --- 模型架構參數 ---
        DEEP_TRANSFORMER_LAYERS = 4
        HIGH_DROPOUT_RATE = 0.2
        DEEP_MLP_HIDDEN_DIMS = [1024, 512, 512, 256]
        HIGHER_MLP_DROPOUT_RATE = 0.4
        
        # --- 1. 分開建立 Bulk (源域) 和 SC (目標域) 的細胞編碼器 ---
        
        # Bulk Encoders (Source Domain)
        self.bulk_gene_tower = GeneTransformerEncoder(
            gene_dim, d_model=256, num_layers=DEEP_TRANSFORMER_LAYERS, dropout=HIGH_DROPOUT_RATE)
        self.bulk_pathway_tower = ResidualMLP(
            pathway_dim, output_dim=256, hidden_dims=DEEP_MLP_HIDDEN_DIMS, dropout_rate=HIGHER_MLP_DROPOUT_RATE)

        # Single-Cell Encoders (Target Domain)
        self.sc_gene_tower = GeneTransformerEncoder(
            gene_dim, d_model=256, num_layers=DEEP_TRANSFORMER_LAYERS, dropout=HIGH_DROPOUT_RATE)
        self.sc_pathway_tower = ResidualMLP(
            pathway_dim, output_dim=256, hidden_dims=DEEP_MLP_HIDDEN_DIMS, dropout_rate=HIGHER_MLP_DROPOUT_RATE)

        # Drug Towers (共享)
        self.graph_tower = GraphTransformer(
            graph_node_dim, embed_dim=256, num_layers=DEEP_TRANSFORMER_LAYERS, dropout=HIGH_DROPOUT_RATE)
        self.fp_tower = ResidualMLP(
            fp_dim, output_dim=256, hidden_dims=DEEP_MLP_HIDDEN_DIMS, dropout_rate=HIGHER_MLP_DROPOUT_RATE)
        
        # 預測器 (共享)
        self.predictor = CrossAttentionFusion(embed_dim=512, nhead=8)

        # ==========================================================
        # --- START OF MODIFICATION: 在這裡加入您的程式碼 ---
        # ==========================================================
        # 為 ADDA 新增域判別器
        feature_dim = 256 # 您的編碼器的輸出維度
        self.gene_discriminator = DomainDiscriminator(input_dim=feature_dim)
        self.pathway_discriminator = DomainDiscriminator(input_dim=feature_dim)
        # ==========================================================
        # --- END OF MODIFICATION ---
        # ==========================================================

    def copy_bulk_to_sc_encoders(self):
        """將預訓練好的 bulk encoder 權重複製到 sc encoder。"""
        print("--- Copying weights from bulk encoders to sc encoders ---")
        self.sc_gene_tower.load_state_dict(self.bulk_gene_tower.state_dict())
        self.sc_pathway_tower.load_state_dict(self.bulk_pathway_tower.state_dict())

    def forward(self, gene_expr, pathway_score, drug_graph=None, drug_fp=None, domain='source'):
        """
        根據 domain 參數選擇使用 bulk 還是 sc 編碼器。
        - 'source': 用於預訓練、微調時的源域數據處理和驗證。
        - 'target': 用於微調時的目標域數據處理和最終預測。
        """
        if domain == 'source':
            gene_feat = self.bulk_gene_tower(gene_expr)
            pathway_feat = self.bulk_pathway_tower(pathway_score)
        elif domain == 'target':
            gene_feat = self.sc_gene_tower(gene_expr)
            pathway_feat = self.sc_pathway_tower(pathway_score)
        else:
            raise ValueError("Domain must be 'source' or 'target'")
            
        # 如果只需要細胞特徵（例如在微調目標域時），直接返回
        if drug_graph is None and drug_fp is None:
            return gene_feat, pathway_feat

        cell_feat = torch.cat([gene_feat, pathway_feat], dim=1)
        
        # 處理藥物特徵
        graph_feat = self.graph_tower(drug_graph)
        fp_feat = self.fp_tower(drug_fp)
        drug_feat = torch.cat([graph_feat, fp_feat], dim=1)
        
        # 進行預測
        prediction = self.predictor(cell_feat, drug_feat)
        
        # 返回所有需要的輸出
        return prediction, gene_feat, pathway_feat

# =============================================================================
# 5. 訓練與評估流程 (已更新)
# =============================================================================
class MultiModalPredictionDataset(Dataset):
    def __init__(self, df_sc_expr, df_sc_pathway, df_smiles, graph_map, fp_map):
        self.df_sc_expr = df_sc_expr
        self.df_sc_pathway = df_sc_pathway
        self.df_smiles = df_smiles
        self.graph_map = graph_map
        self.fp_map = fp_map

        self.samples = []
        for cell_name in self.df_sc_expr.index:
            for drug_name in self.df_smiles.index:
                self.samples.append((cell_name, drug_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cell_name, drug_name = self.samples[idx]

        gene_expr = torch.FloatTensor(self.df_sc_expr.loc[cell_name].values)
        pathway_score = torch.FloatTensor(self.df_sc_pathway.loc[cell_name].values)
        drug_graph = self.graph_map[drug_name]
        drug_fp = self.fp_map[drug_name]

        return gene_expr, pathway_score, drug_graph, drug_fp, cell_name, drug_name


# 新增：適應四輸入模型的 Collate Function for Prediction
def custom_collate_pred_fn(batch):
    gene_exprs, pathway_scores, drug_graphs, drug_fps, cell_names, drug_names = zip(*batch)
    return torch.stack(gene_exprs, 0), torch.stack(pathway_scores, 0), GraphDataLoader(list(drug_graphs),
                                                                                       batch_size=len(
                                                                                           drug_graphs)), torch.stack(
        drug_fps, 0), cell_names, drug_names


def evaluate(model, dataloader, device, domain='source'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (gene_expr, pathway_score, drug_graph_loader, drug_fp, labels) in enumerate(dataloader):
            drug_graph_batch = next(iter(drug_graph_loader)).to(device)
            gene_expr, pathway_score, drug_fp, labels = gene_expr.to(device), pathway_score.to(device), drug_fp.to(device), labels.to(device)
            
            # 傳遞 domain 參數
            preds, _, _ = model(gene_expr, pathway_score, drug_graph_batch, drug_fp, domain=domain)
            
            all_preds.extend(torch.softmax(preds, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    model.train()
    if len(np.unique(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_preds)



# --- 修改 perform_inference_and_save 函數以使用 sc_encoder ---
def perform_inference_and_save(model, pred_loader, device, output_path):
    print(f"\n--- Performing inference on single-cell data using SC-ENCODERS ---")
    model.eval()
    predictions_dict = {}
    with torch.no_grad():
        for i, (gene_expr, pathway_score, drug_graph_loader, drug_fp, cell_names, drug_names) in enumerate(pred_loader):
            drug_graph_batch = next(iter(drug_graph_loader)).to(device)
            gene_expr, pathway_score, drug_fp = gene_expr.to(device), pathway_score.to(device), drug_fp.to(device)
            
            # 關鍵修改：使用 domain='target' 來確保調用 sc_encoders
            preds, _, _ = model(gene_expr, pathway_score, drug_graph_batch, drug_fp, domain='target')

            probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()
            for cell, drug, prob in zip(cell_names, drug_names, probs):
                if cell not in predictions_dict:
                    predictions_dict[cell] = {}
                predictions_dict[cell][drug] = prob
            
            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} batches...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results_df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    results_df.to_csv(output_path)
    print(f"--- Inference finished. Prediction probabilities saved to {output_path} ---")


# 新增：第一階段 - 監督式預訓練函數
def train_source_pretraining(model, train_loader, val_loader, optimizer, scheduler, config):
    checkpoint_dir = "/home/bio_wangxf/jupyterlab/GSE140440_du145/checkpoints_test/pretrain_advanced_gdsc1.3"
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    best_auc_ckpt_path = os.path.join(checkpoint_dir, "best_auc_checkpoint.pth")

    # --- START OF MODIFICATION ---
    # 使用標籤平滑 (Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # --- END OF MODIFICATION ---

    start_epoch, best_val_auc = 0, 0.0

    if os.path.exists(last_ckpt_path):
        print("--- Resuming pre-training from checkpoint ---")
        checkpoint = torch.load(last_ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint.get('best_val_auc', 0.0)

    print("\n--- Starting Phase 1: Supervised Pre-training on Source Domain ---")
    for epoch in range(start_epoch, config.PRETRAIN_EPOCHS):
        model.train()
        total_loss = 0
        for gene_expr, pathway_score, drug_graph_loader, drug_fp, labels in train_loader:
            gene_expr, pathway_score, drug_fp, labels = gene_expr.to(config.DEVICE), pathway_score.to(
                config.DEVICE), drug_fp.to(config.DEVICE), labels.to(config.DEVICE)

            # 从 PyG DataLoader 中获取批次数据，并将其移到设备上
            # 这里是核心修改部分
            drug_graph_batch = next(iter(drug_graph_loader)).to(config.DEVICE)

            optimizer.zero_grad()
            
            # 確保使用 bulk encoders
            prediction, _, _ = model(gene_expr, pathway_score, drug_graph_batch, drug_fp, domain='source')

            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_auc = evaluate(model, val_loader, config.DEVICE, domain='source')
        scheduler.step(val_auc)
        print(
            f"Pre-train Epoch {epoch + 1}/{config.PRETRAIN_EPOCHS} | Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, best_auc_ckpt_path)
            print(f"  * New best validation AUC: {best_val_auc:.4f}. Pre-trained model saved.")
        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'best_val_auc': best_val_auc}, last_ckpt_path)

    print("--- Phase 1: Pre-training Finished ---")
    return best_auc_ckpt_path


# 修改：第二階段 - 領域自適應微調函數
# 修改：第二階段 - 領域自適應微調函數
# 修改：第二階段 - 領域自適應微調函數
def train_domain_adaptation(model, source_loader, target_loader, val_loader, optimizer, scheduler, adaptation_method, config):
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoints", "ADDA")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    
    # --- MODIFICATION: Define paths for the two best models ---
    best_T_loss_ckpt_path = os.path.join(checkpoint_dir, "best_T_loss_model_checkpoint.pth")
    best_D_loss_ckpt_path = os.path.join(checkpoint_dir, "best_D_loss_model_checkpoint.pth")
    
    # --- MODIFICATION: Initialize trackers for both losses ---
    start_epoch = 0
    best_T_loss = float('inf') # We want to minimize this
    best_D_loss = float('-inf') # We want to MAXIMIZE this

    # 1. 複製預訓練權重
    model.copy_bulk_to_sc_encoders()

    # 2. 凍結所有非目標域 (SC) 編碼器的部分
    print("--- Freezing ALL modules except SC-Encoders & Discriminators for ADDA fine-tuning ---")
    for param in model.bulk_gene_tower.parameters(): param.requires_grad = False
    for param in model.bulk_pathway_tower.parameters(): param.requires_grad = False
    for param in model.graph_tower.parameters(): param.requires_grad = False
    for param in model.fp_tower.parameters(): param.requires_grad = False
    for param in model.predictor.parameters(): param.requires_grad = False

    # 3. 建立兩個獨立的優化器
    optimizer_T = optim.AdamW(
        list(model.sc_gene_tower.parameters()) + list(model.sc_pathway_tower.parameters()),
        lr=config.LR_FINETUNE, weight_decay=config.WEIGHT_DECAY
    )
    optimizer_D = optim.AdamW(
        list(model.gene_discriminator.parameters()) + list(model.pathway_discriminator.parameters()),
        lr=config.LR_FINETUNE, weight_decay=config.WEIGHT_DECAY
    )
    criterion_D = nn.BCEWithLogitsLoss()

    # (模型恢復邏輯可以簡化或移除)

    print(f"\n--- Starting Phase 2: Fine-tuning with ADDA ---")
    print("--- Training SC-ENCODERS to fool Domain Discriminators. ---")
    
    len_dataloader = min(len(source_loader), len(target_loader))

    for epoch in range(start_epoch, config.FINETUNE_EPOCHS):
        model.train()
        
        total_D_loss, total_T_loss = 0, 0
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for i in range(len_dataloader):
            s_gene, s_path, _, _, _ = next(source_iter)
            t_gene, t_path = next(target_iter)
            s_gene, s_path = s_gene.to(config.DEVICE), s_path.to(config.DEVICE)
            t_gene, t_path = t_gene.to(config.DEVICE), t_path.to(config.DEVICE)

            # === Train Domain Discriminators (Step 1) ===
            for param in model.gene_discriminator.parameters(): param.requires_grad = True
            for param in model.pathway_discriminator.parameters(): param.requires_grad = True
            optimizer_D.zero_grad()
            
            with torch.no_grad():
                source_gene_feat, source_pathway_feat = model(s_gene, s_path, domain='source', drug_graph=None, drug_fp=None)
                target_gene_feat, target_pathway_feat = model(t_gene, t_path, domain='target', drug_graph=None, drug_fp=None)
            
            d_source_gene = model.gene_discriminator(source_gene_feat)
            d_target_gene = model.gene_discriminator(target_gene_feat)
            d_source_path = model.pathway_discriminator(source_pathway_feat)
            d_target_path = model.pathway_discriminator(target_pathway_feat)

            domain_source = torch.zeros(s_gene.size(0), 1).to(config.DEVICE)
            domain_target = torch.ones(t_gene.size(0), 1).to(config.DEVICE)

            loss_D = (criterion_D(d_source_gene, domain_source) + criterion_D(d_target_gene, domain_target) +
                      criterion_D(d_source_path, domain_source) + criterion_D(d_target_path, domain_target))
            
            loss_D.backward()
            optimizer_D.step()
            total_D_loss += loss_D.item()

            # === Train Target Encoders (Step 2) ===
            for param in model.gene_discriminator.parameters(): param.requires_grad = False
            for param in model.pathway_discriminator.parameters(): param.requires_grad = False
            optimizer_T.zero_grad()
            
            target_gene_feat, target_pathway_feat = model(t_gene, t_path, domain='target', drug_graph=None, drug_fp=None)
            
            d_target_gene_adv = model.gene_discriminator(target_gene_feat)
            d_target_path_adv = model.pathway_discriminator(target_pathway_feat)
            
            loss_T = (criterion_D(d_target_gene_adv, domain_source) + 
                      criterion_D(d_target_path_adv, domain_source))
            
            loss_T.backward()
            optimizer_T.step()
            total_T_loss += loss_T.item()

        avg_D_loss = total_D_loss / len_dataloader
        avg_T_loss = total_T_loss / len_dataloader

        print(f"Fine-tune Epoch {epoch + 1}/{config.FINETUNE_EPOCHS} | "
              f"Discriminator Loss (D_loss): {avg_D_loss:.4f} | "
              f"Target Encoder Loss (T_loss): {avg_T_loss:.4f} | "
              f"LR: {optimizer_T.param_groups[0]['lr']:.2e}")
        
        # --- MODIFICATION: Save checkpoints based on two different criteria ---
        
        # 1. Save model with the best (lowest) Target Encoder Loss
        if avg_T_loss < best_T_loss:
            best_T_loss = avg_T_loss
            torch.save({'model_state_dict': model.state_dict()}, best_T_loss_ckpt_path)
            print(f"  * New best Target Encoder Loss: {best_T_loss:.4f}. Checkpoint saved to best_T_loss_model.")

        # 2. Save model with the best (highest) Discriminator Loss
        if avg_D_loss > best_D_loss:
            best_D_loss = avg_D_loss
            torch.save({'model_state_dict': model.state_dict()}, best_D_loss_ckpt_path)
            print(f"  * New best (highest) Discriminator Loss: {best_D_loss:.4f}. Checkpoint saved to best_D_loss_model.")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, last_ckpt_path)

    # Unfreeze all layers
    for p in model.parameters(): p.requires_grad = True
    print(f"--- Phase 2: Fine-tuning Finished for ADDA ---")
    
    # Return the paths to the two saved models
    return best_T_loss_ckpt_path, best_D_loss_ckpt_path



# =============================================================================
# 6. 主執行流程
# =============================================================================
if __name__ == '__main__':
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    train_df, val_df, df_bulk_expr, df_sc_expr, df_bulk_pathway, df_sc_pathway, df_smiles, common_genes, common_pathways = load_and_prepare_data(
        cfg)

    print("\n--- Pre-calculating all drug features (Graphs and Fingerprints) ---")
    graph_map = {name: smiles_to_graph(s) for name, s in df_smiles['smiles'].items()}
    fp_map = {name: smiles_to_fingerprint(s, n_bits=cfg.FP_SIZE) for name, s in df_smiles['smiles'].items()}

    # 數據加載器
    num_cpu_workers = 0
    source_train_loader = DataLoader(MultiModalDataset(train_df, df_bulk_expr, df_bulk_pathway, graph_map, fp_map),
                                     batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                                     num_workers=num_cpu_workers, drop_last=True)
    source_val_loader = DataLoader(MultiModalDataset(val_df, df_bulk_expr, df_bulk_pathway, graph_map, fp_map),
                                   batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn,
                                   num_workers=num_cpu_workers, drop_last=True)
    target_loader = DataLoader(SCTargetDataset(df_sc_expr, df_sc_pathway),
                               batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=num_cpu_workers, drop_last=True)

    # --- 動態獲取模型輸入維度（使用 PCA 後的基因維度） ---
    gene_dim = df_bulk_expr.shape[1]   #  這裡改成 PCA 後的維度，例如 256
    pathway_dim = df_bulk_pathway.shape[1]
    fp_dim = cfg.FP_SIZE
    graph_node_dim = 33  # 根據 smiles_to_graph 函數

    # --- Phase 1: 監督式預訓練 ---
    pretrain_checkpoint_path = "/home/bio_wangxf/jupyterlab/GSE140440_du145/checkpoints_test/pretrain_advanced_gdsc1.3/best_auc_checkpoint.pth"
    if not os.path.exists(pretrain_checkpoint_path):
        pretrain_model = FourTowerModel(gene_dim, pathway_dim, fp_dim, graph_node_dim).to(cfg.DEVICE)
        optimizer_pretrain = optim.AdamW(pretrain_model.parameters(), lr=cfg.LR_PRETRAIN, weight_decay=cfg.WEIGHT_DECAY)
        scheduler_pretrain = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_pretrain, 'max', factor=0.5, patience=5, verbose=True
        )
        train_source_pretraining(pretrain_model, source_train_loader, source_val_loader,
                                 optimizer_pretrain, scheduler_pretrain, cfg)
    else:
        print("--- Found existing pre-trained model. Skipping Phase 1. ---")

    # --- Phase 2: 領域自適應微調 + 即時推理 ---
    # --- MODIFICATION: Only run for CORAL ---
    for method in ["ADDA"]:
        print(f"\n=== Fine-tuning with {method} ===")
        finetune_model = FourTowerModel(gene_dim, pathway_dim, fp_dim, graph_node_dim).to(cfg.DEVICE)

        # 加載預訓練權重
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=cfg.DEVICE)
        finetune_model.load_state_dict(checkpoint['model_state_dict'])

        # optimizer 和 scheduler 會在 train_domain_adaptation 內部被重新建立，所以這裡傳入的只是佔位符
        optimizer_finetune = None
        scheduler_finetune = None

        # 微調，返回最佳 checkpoint 路徑
        _, best_loss_path = train_domain_adaptation(
            finetune_model, source_train_loader, target_loader,
            source_val_loader, optimizer_finetune, scheduler_finetune,
            method, cfg
        )

        # === 每個方法完成後立刻做推理並保存 ===
        pred_dataset = MultiModalPredictionDataset(df_sc_expr, df_sc_pathway, df_smiles, graph_map, fp_map)
        pred_loader = DataLoader(pred_dataset, batch_size=cfg.BATCH_SIZE * 2,
                                 shuffle=False, collate_fn=custom_collate_pred_fn,
                                 num_workers=num_cpu_workers)

        # 只使用 best_loss 模型進行預測
        model_type = "best_loss_model"
        model_path = best_loss_path
        
        print(f"\n--- Performing inference using {method}'s {model_type} model ---")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint file not found at {model_path}. Skipping.")
            continue

        inf_model = FourTowerModel(gene_dim, pathway_dim, fp_dim, graph_node_dim).to(cfg.DEVICE)
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
        inf_model.load_state_dict(checkpoint['model_state_dict'])

        output_filename = os.path.join(cfg.OUTPUT_DIR, f"sc_drug_predictions_{method}_{model_type}.csv")
        perform_inference_and_save(inf_model, pred_loader, cfg.DEVICE, output_path=output_filename)

    print("\n=== All fine-tuning and inference complete! ===")


