import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class CCSNet(torch.nn.Module):
    """
    Carrier-Centric Spatial Network (CCSNet)
    A robust Graph Attention Network designed to predict possession survival 
    based on localized spatiotemporal pressure graphs.
    """
    def __init__(self, node_dim=5, edge_dim=4, hidden_dim=64, heads=4, dropout=0.3):
        super(CCSNet, self).__init__()
        
        self.dropout_rate = dropout
        
        # 1. Input Encoders
        # Projects our raw physical features into a higher-dimensional learning space
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # 2. Graph Attention Layers (GATv2)
        # We use multi-head attention (like Transformers) to let the network look at 
        # the graph from different tactical perspectives simultaneously.
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 3. Output MLP (The Decision Engine)
        # Takes the pooled graph representation and predicts the probability of keeping the ball
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # *2 because we concatenate Mean and Max pooling
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(hidden_dim // 2, 1) # Single output logit (Survival Probability)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Encode Nodes
        x = self.node_encoder(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # 2. First Attention Block with Residual Connection
        # Residuals prevent the network from "forgetting" the original node identity
        residual = x
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x + residual 
        
        # 3. Second Attention Block
        residual = x
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x + residual
        
        # 4. Global Graph Pooling (Compressing the tactical map)
        # EDGE CASE FIX: We use BOTH Max Pooling (detects extreme single-player pressure)
        # and Mean Pooling (detects overall structural crowding)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        
        # Concatenate them into one holistic tactical summary
        graph_embedding = torch.cat([x_max, x_mean], dim=1)
        
        # 5. Final Prediction
        out = self.mlp(graph_embedding)
        
        # We return raw logits. We will apply Sigmoid() during the loss calculation
        # using BCEWithLogitsLoss (which is much more numerically stable).
        return out

# ==========================================
# SANITY CHECK PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Testing CCSNet Architecture robustness with dummy data...")
    from torch_geometric.data import Data, Batch
    
    # Create a fake graph (1 Carrier, 3 Defenders)
    # 5 Node Features: X, Y, Vx, Vy, Is_Teammate
    dummy_x = torch.rand((4, 5)) 
    
    # Edge Index: 3 surrounding players pointing to Node 0 (Carrier)
    dummy_edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.long)
    
    # 4 Edge Features: Distance, TTI, Closing Speed, Is_Teammate
    dummy_edge_attr = torch.rand((3, 4))
    
    # Batch index (all 4 nodes belong to Graph 0)
    dummy_batch = torch.zeros(4, dtype=torch.long)
    
    model = CCSNet()
    
    # Forward Pass
    try:
        output = model(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch)
        prob = torch.sigmoid(output).item()
        print("Architecture built successfully!")
        print(f"Forward pass survived. Shape: {output.shape}")
        print(f"Sample Survival Probability Output: {prob:.4f}")
    except Exception as e:
        print(f"Architecture Failed: {e}")