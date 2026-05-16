import torch
from torch_geometric.data import Data
import pickle
import os
from tqdm import tqdm

class CCSNDatasetLoader:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path

    def process_to_tensors(self, output_pt_path):
        print(f"Loading raw graph dictionaries from {self.pkl_path}...")
        
        # Verify file exists before trying to open it
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Could not find {self.pkl_path}. Did you run the builder script?")
            
        with open(self.pkl_path, 'rb') as f:
            raw_graphs = pickle.load(f)
            
        pyg_data_list = []
        skipped_graphs = 0
        
        print("Converting to PyTorch Geometric Tensors...")
        
        for graph in tqdm(raw_graphs):
            # ==========================================
            # DEFENSIVE CHECK 1: The Missing Label
            # ==========================================
            if 'label' not in graph:
                skipped_graphs += 1
                continue
            
            # 1. NODE FEATURES (X)
            # Ensure all values are strictly floats (PyTorch hates mixed types)
            x = []
            node_id_to_idx = {} 
            
            for idx, node in enumerate(graph['nodes']):
                x.append([
                    float(node['pos'][0]),
                    float(node['pos'][1]),
                    float(node['vel'][0]),
                    float(node['vel'][1]),
                    float(node['is_teammate'])
                ])
                node_id_to_idx[node['id']] = idx
                
            x_tensor = torch.tensor(x, dtype=torch.float)
            
            # 2. EDGE CONNECTIVITY (edge_index) & EDGE FEATURES (edge_attr)
            edge_sources = []
            edge_targets = []
            edge_attr = []
            
            for edge in graph['edges']:
                # DEFENSIVE CHECK 2: Edge Node Verification
                if edge['source'] not in node_id_to_idx or edge['target'] not in node_id_to_idx:
                    continue
                    
                src_idx = node_id_to_idx[edge['source']]
                tgt_idx = node_id_to_idx[edge['target']]
                
                edge_sources.append(src_idx)
                edge_targets.append(tgt_idx)
                
                # Sanitize infinite TTI values down to a max of 10 seconds
                tti = edge['tti'] if edge['tti'] != float('inf') else 10.0 
                
                edge_attr.append([
                    float(edge['distance']),
                    float(tti),
                    float(edge['closing_speed']),
                    float(edge['is_teammate'])
                ])
            
            # DEFENSIVE CHECK 3: Empty Edges
            # If a player has the ball but literally no one is around them within our TTI limits
            if len(edge_sources) == 0:
                edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
                edge_attr_tensor = torch.empty((0, 4), dtype=torch.float)
            else:
                edge_index_tensor = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
                edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
            
            # 3. TARGET LABEL (y)
            y_tensor = torch.tensor([graph['label']], dtype=torch.float)
            
            # 4. Construct the PyG Data Object
            data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor)
            pyg_data_list.append(data)
            
        print(f"\nSuccessfully converted {len(pyg_data_list)} graphs to Tensors.")
        
        if skipped_graphs > 0:
            print(f"⚠️ WARNING: Skipped {skipped_graphs} corrupted graphs (Missing 'label').")
            print("If this number is huge, you MUST re-run ccsn_builder.py to overwrite the old .pkl file!")
            
        print(f"Saving PyTorch Dataset to {output_pt_path}...")
        os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
        torch.save(pyg_data_list, output_pt_path)
        print("Done! Ready for Neural Network Training.")

if __name__ == "__main__":
    loader = CCSNDatasetLoader("../../data/processed/ccsn_master_dataset.pkl")
    loader.process_to_tensors("../../data/processed/ccsn_tensors.pt")