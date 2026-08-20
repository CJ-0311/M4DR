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
from sklearn.preprocessing import StandardScaler
import math
import warnings
import sys
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
module_path = "~/jupyterlab/scdeal"
if module_path not in sys.path:
    sys.path.append(module_path)

# 确保您的 loss.py 和 mmd.py 位于Python路径中
# 假设 loss.py 和 mmd.py 位于同一目录
import loss as custom_loss
import mmd as custom_mmd

# =============================================================================
# 0. 配置参数 (Configuration)
# =============================================================================
class Config:
    # --- 文件路径 ---
    BULK_EXPR_PATH = '~/jupyterlab/hvg4000_allbulk4.csv'
    DRUG_SMILES_PATH = '~/jupyterlab/alldrugname_with_smiles.csv'
    DRUG_RESPONSE_PATH = '~/jupyterlab/gdsc1.3.csv'
    SINGLE_CELL_EXPR_PATH = '~/jupyterlab/GSE140440pc3tpm.csv'
    BULK_PATHWAY_PATH = '~/jupyterlab/bulk_pathwayscoreMatrix.csv'
    SC_PATHWAY_PATH = '~/jupyterlab/GSE140441_pathwayscoreMatrix.csv'

    # --- 训练超参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    FP_SIZE = 2048  # 摩根指纹维度
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

# =============================================================================
# 1. 数据加载和预处理 (与之前相同)
# =============================================================================
def load_and_prepare_data(config):
    print("--- Starting Data Loading and Preprocessing (Advanced) ---")
    # 加载所有数据
    df_bulk_expr = pd.read_csv(config.BULK_EXPR_PATH, index_col=0)
    df_smiles = pd.read_csv(config.DRUG_SMILES_PATH, usecols=['alldrugname', 'smiles']).dropna()
    
    # 在读取 CSV 时，将 'NA', 'N/A' 等常见缺失值字符串直接转换为 np.nan
    na_values = ['NA', 'N/A', 'NaN', 'nan', '']
    df_response = pd.read_csv(config.DRUG_RESPONSE_PATH, index_col=0, na_values=na_values)

    df_sc_expr = pd.read_csv(config.SINGLE_CELL_EXPR_PATH, index_col=0)
    df_bulk_pathway = pd.read_csv(config.BULK_PATHWAY_PATH, index_col=0)
    df_sc_pathway = pd.read_csv(config.SC_PATHWAY_PATH, index_col=0)

    # 预处理 (转置，重命名等)
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

    # --- 数据对齐 ---
    common_drugs = sorted(list(set(df_smiles['alldrugname']) & set(df_response.columns)))
    common_bulk_cells = sorted(list(set(df_bulk_expr.index) & set(df_response.index) & set(df_bulk_pathway.index)))
    common_genes = sorted(list(set(df_bulk_expr.columns) & set(df_sc_expr.columns)))
    common_pathways = sorted(list(set(df_bulk_pathway.columns) & set(df_sc_pathway.columns)))

    print(f"Number of common drugs: {len(common_drugs)}")
    print(f"Number of common bulk cells: {len(common_bulk_cells)}")
    print(f"Number of common genes: {len(common_genes)}")
    print(f"Number of common pathways: {len(common_pathways)}")
    
    # 过滤数据
    df_smiles = df_smiles[df_smiles['alldrugname'].isin(common_drugs)].drop_duplicates(subset=['alldrugname']).set_index('alldrugname').loc[common_drugs]
    df_response = df_response.loc[common_bulk_cells, common_drugs]
    df_bulk_expr = df_bulk_expr.loc[common_bulk_cells, common_genes]
    df_sc_expr = df_sc_expr[common_genes]
    df_bulk_pathway = df_bulk_pathway.loc[common_bulk_cells, common_pathways]
    df_sc_pathway = df_sc_pathway[common_pathways]

    # --- 特征缩放 ---
    df_sc_expr = np.log1p(df_sc_expr)
    scaler_gene = StandardScaler()
    df_bulk_expr[:] = scaler_gene.fit_transform(df_bulk_expr)
    df_sc_expr[:] = scaler_gene.transform(df_sc_expr)

    scaler_pathway = StandardScaler()
    df_bulk_pathway[:] = scaler_pathway.fit_transform(df_bulk_pathway)
    df_sc_pathway[:] = scaler_pathway.transform(df_sc_pathway)

    print("--- Data Loading and Preprocessing Finished ---")
    return df_bulk_expr, df_sc_expr, df_bulk_pathway, df_sc_pathway, df_smiles, common_genes, common_pathways

# =============================================================================
# 2. 特征生成与数据集类 (已更新)
# =============================================================================
def smiles_to_graph(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    atom_features = [[atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetHybridization()),
                      int(atom.GetIsAromatic())] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0),
                                                                                                                dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def smiles_to_fingerprint(smiles_string, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return torch.zeros(n_bits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)

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

# 适应四输入模型的 Collate Function for Prediction
def custom_collate_pred_fn(batch):
    gene_exprs, pathway_scores, drug_graphs, drug_fps, cell_names, drug_names = zip(*batch)
    return torch.stack(gene_exprs, 0), torch.stack(pathway_scores, 0), GraphDataLoader(list(drug_graphs),
                                                                                       batch_size=len(
                                                                                           drug_graphs)), torch.stack(
        drug_fps, 0), cell_names, drug_names

# =============================================================================
# 3. 模型架构 (与训练时相同)
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
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 残差连接
        return self.relu(x + self.block(x))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[1024, 512, 512, 256], dropout_rate=0.4):
        super().__init__()

        # 输入层
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rate)
        ]

        # 残差块
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout_rate))
            else:
                # 如果维度变化，则使用标准的全连接层
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        # 关键：batch_first=True 使得输入维度为 (batch, seq, feature)
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
        # 关键：传入 key_padding_mask 告知注意力层忽略padding部分
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class GraphTransformer(nn.Module):
    # 修改 __init__ 签名以接受 dropout
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        # 将 dropout 传递给每一层
        self.layers = nn.ModuleList([GraphTransformerLayer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def to_dense_batch(self, x, batch):
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = torch.bincount(batch)
        max_nodes = num_nodes_per_graph.max().item()

        # 创建一个填充好的张量和一个mask
        x_padded = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
        # key_padding_mask: True代表是padding，需要被忽略
        key_padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device)

        # 填充数据和mask
        for i in range(batch_size):
            nodes = x[batch == i]
            num_nodes = len(nodes)
            x_padded[i, :num_nodes] = nodes
            key_padding_mask[i, :num_nodes] = False  # False代表是真实数据

        return x_padded, key_padding_mask

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.input_proj(x)

        # 1. 将稀疏的图批次转换为密集(padded)的序列批次和对应的padding mask
        x_padded, key_padding_mask = self.to_dense_batch(x, batch)

        # 2. 将padded序列和mask传入Transformer层
        for layer in self.layers:
            x_padded = layer(x_padded, key_padding_mask=key_padding_mask)

        # 3. 在池化前，移除padding的影响
        # 我们只对有效的节点进行池化
        # key_padding_mask中False代表有效节点，所以用~反转
        valid_nodes_mask = ~key_padding_mask.squeeze(-1) if key_padding_mask.dim() == 3 else ~key_padding_mask
        x = x_padded[valid_nodes_mask]

        # 4. 全局池化得到图的表示
        x = global_mean_pool(x, batch)
        x = self.output_proj(x)
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, nhead=4, dropout=0.1):
        super().__init__()
        # 注意力层，让cell特征作为query，drug特征作为key/value
        self.cell_drug_attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        # 注意力层，让drug特征作为query，cell特征作为key/value
        self.drug_cell_attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 将交互后的特征融合
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, cell_feat, drug_feat):
        # (Batch, Dim) -> (Batch, 1, Dim) for attention
        cell_feat_unsq = cell_feat.unsqueeze(1)
        drug_feat_unsq = drug_feat.unsqueeze(1)

        # cell "关注" drug
        cell_enhanced, _ = self.cell_drug_attention(cell_feat_unsq, drug_feat_unsq, drug_feat_unsq)
        # Squeeze, add residual connection, and normalize
        cell_enhanced = self.norm1(cell_enhanced.squeeze(1) + cell_feat)

        # drug "关注" cell
        drug_enhanced, _ = self.drug_cell_attention(drug_feat_unsq, cell_feat_unsq, cell_feat_unsq)
        # Squeeze, add residual connection, and normalize
        drug_enhanced = self.norm2(drug_enhanced.squeeze(1) + drug_feat)

        # 拼接增强后的特征
        final_feat = torch.cat([cell_enhanced, drug_enhanced], dim=1)
        return self.output_mlp(final_feat)

class FourTowerModel(nn.Module):
    def __init__(self, gene_dim, pathway_dim, fp_dim, graph_node_dim):
        super().__init__()

        # 增加 Transformer 层数和 Dropout
        DEEP_TRANSFORMER_LAYERS = 4
        HIGH_DROPOUT_RATE = 0.2

        # 增加 MLP 深度和 Dropout
        DEEP_MLP_HIDDEN_DIMS = [1024, 512, 512, 256]
        HIGHER_MLP_DROPOUT_RATE = 0.4

        # 四塔 (Towers are now deeper and have higher dropout)
        self.gene_tower = GeneTransformerEncoder(
            gene_dim,
            d_model=256,
            num_layers=DEEP_TRANSFORMER_LAYERS,
            dropout=HIGH_DROPOUT_RATE
        )

        # 使用升级后的 ResidualMLP
        self.pathway_tower = ResidualMLP(
            pathway_dim,
            output_dim=256,
            hidden_dims=DEEP_MLP_HIDDEN_DIMS,
            dropout_rate=HIGHER_MLP_DROPOUT_RATE
        )

        self.graph_tower = GraphTransformer(
            graph_node_dim,
            embed_dim=256,
            num_layers=DEEP_TRANSFORMER_LAYERS,
            dropout=HIGH_DROPOUT_RATE  # 传入更高的 dropout
        )

        # 使用升级后的 ResidualMLP
        self.fp_tower = ResidualMLP(
            fp_dim,
            output_dim=256,
            hidden_dims=DEEP_MLP_HIDDEN_DIMS,
            dropout_rate=HIGHER_MLP_DROPOUT_RATE
        )

        # 融合与预测器 (Fusion and Predictor - Unchanged from previous version)
        self.predictor = CrossAttentionFusion(embed_dim=512, nhead=8)

    def forward(self, gene_expr, pathway_score, drug_graph=None, drug_fp=None, return_features='all'):
        gene_feat = self.gene_tower(gene_expr)
        pathway_feat = self.pathway_tower(pathway_score)
        cell_feat = torch.cat([gene_feat, pathway_feat], dim=1)

        if return_features == 'cell':
            return cell_feat

        graph_feat = self.graph_tower(drug_graph)
        fp_feat = self.fp_tower(drug_fp)
        drug_feat = torch.cat([graph_feat, fp_feat], dim=1)

        if return_features == 'all':
            prediction = self.predictor(cell_feat, drug_feat)
            return prediction, cell_feat, drug_feat

        return None

# =============================================================================
# 4. 预测函数
# =============================================================================
def perform_inference_and_save(model, pred_loader, device, output_path):
    print(f"\n--- Performing inference on single-cell data ---")
    model.eval()
    predictions_dict = {}

    with torch.no_grad():
        for i, (gene_expr, pathway_score, drug_graph_loader, drug_fp, cell_names, drug_names) in enumerate(pred_loader):
            
            # 正确处理 GraphDataLoader：从中提取批次数据并移动到 device
            drug_graph_batch = next(iter(drug_graph_loader)).to(device)

            # 将其他张量移动到 device
            gene_expr, pathway_score, drug_fp = gene_expr.to(device), pathway_score.to(
                device), drug_fp.to(device)

            # 将正确的批次数据传递给模型
            preds, _, _ = model(gene_expr, pathway_score, drug_graph_batch, drug_fp)

            # 这里我们保存概率值，而不是0/1的分类结果，信息量更丰富
            probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()

            for cell, drug, prob in zip(cell_names, drug_names, probs):
                if cell not in predictions_dict:
                    predictions_dict[cell] = {}
                predictions_dict[cell][drug] = prob

            if (i + 1) % pred_loader.batch_size == 0 or (i + 1) == len(pred_loader):
                processed_samples = (i + 1) * pred_loader.batch_size
                if processed_samples > len(pred_loader.dataset):
                    processed_samples = len(pred_loader.dataset)
                print(f"  Processed {processed_samples} / {len(pred_loader.dataset)} samples...")

    results_df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    results_df.to_csv(output_path)
    print(f"--- Inference finished. Prediction probabilities saved to {output_path} ---")

# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Predict drug response using trained models')
    parser.add_argument('--method', type=str, required=True, choices=['DAN', 'JAN', 'MMD'], 
                        help='Domain adaptation method to use')
    parser.add_argument('--checkpoint_type', type=str, required=True, choices=['best_auc', 'best_loss'], 
                        help='Type of checkpoint to use')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                        help='Output file path for predictions')
    
    args = parser.parse_args()
    
    cfg = Config()
    
    # 加载数据
    df_bulk_expr, df_sc_expr, df_bulk_pathway, df_sc_pathway, df_smiles, common_genes, common_pathways = load_and_prepare_data(cfg)
    
    # 预计算所有药物特征
    print("\n--- Pre-calculating all drug features (Graphs and Fingerprints) ---")
    graph_map = {name: smiles_to_graph(s) for name, s in df_smiles['smiles'].items()}
    fp_map = {name: smiles_to_fingerprint(s, n_bits=cfg.FP_SIZE) for name, s in df_smiles['smiles'].items()}
    
    # 构建预测数据集和数据加载器
    pred_dataset = MultiModalPredictionDataset(df_sc_expr, df_sc_pathway, df_smiles, graph_map, fp_map)
    pred_loader = DataLoader(pred_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                             collate_fn=custom_collate_pred_fn, num_workers=4)
    
    # 动态获取模型输入维度
    gene_dim = len(common_genes)
    pathway_dim = len(common_pathways)
    fp_dim = cfg.FP_SIZE
    graph_node_dim = 5  # 根据smiles_to_graph函数
    
    # 构建模型
    model = FourTowerModel(gene_dim, pathway_dim, fp_dim, graph_node_dim).to(cfg.DEVICE)
    
    # 加载模型权重
    model_path = f"/home/bio_wangxf/jupyterlab/lam0.1/checkpoints_test_advanced_gdsc1.3/{args.method}/{args.checkpoint_type}_checkpoint.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 进行预测
    perform_inference_and_save(model, pred_loader, cfg.DEVICE, args.output)

if __name__ == '__main__':
    main()