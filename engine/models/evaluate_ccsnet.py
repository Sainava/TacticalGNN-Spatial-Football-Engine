import torch
import pickle
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, brier_score_loss
from tqdm import tqdm

# Import your custom architecture
from gnn_architecture import CCSNet

def evaluate_model():
    # 1. HARDWARE SETUP
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Executing Inference on {device}...")

    # 2. LOAD FILES
    print("Loading Tensor Dataset and Raw Dictionaries...")
    pt_path = "../../data/processed/ccsn_tensors.pt"
    pkl_path = "../../data/processed/ccsn_master_dataset.pkl"
    model_path = "../../models/ccsnet_best_model.pth"
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return

    # Load Tensors and Raw Data
    dataset = torch.load(pt_path, weights_only=False)
    with open(pkl_path, 'rb') as f:
        raw_graphs = pickle.load(f)
        
    print(f"Loaded {len(dataset)} graphs for evaluation.")

    # 3. INITIALIZE MODEL
    model = CCSNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 4. INFERENCE LOOP
    print("Running global inference and mapping predictions to frames...")
    
    y_true = []
    y_prob = []
    detailed_results = []
    
    with torch.no_grad():
        # We zip the raw data and tensor data together to keep the frame IDs attached
        for raw_graph, pyg_graph in tqdm(zip(raw_graphs, dataset), total=len(dataset)):
            
            # Move data to GPU/CPU
            x = pyg_graph.x.to(device)
            edge_index = pyg_graph.edge_index.to(device)
            edge_attr = pyg_graph.edge_attr.to(device)
            
            # Since we are predicting 1 graph at a time, the batch index is just zeros for all nodes
            batch_idx = torch.zeros(x.size(0), dtype=torch.long).to(device)
            
            # Forward pass
            out = model(x, edge_index, edge_attr, batch_idx)
            prob = torch.sigmoid(out).item()
            actual_label = pyg_graph.y.item()
            
            y_true.append(actual_label)
            y_prob.append(prob)
            
            # Store everything for error analysis
            detailed_results.append({
                'frame': raw_graph['frame'],
                'team': raw_graph['carrier_team'],
                'carrier': raw_graph['carrier_id'],
                'actual': actual_label,
                'predicted_prob': prob
            })

    # 5. CALCULATE ADVANCED METRICS
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob) # Closer to 0 is better

    print("\n=============================================")
    print(" GLOBAL DIAGNOSTIC METRICS")
    print("=============================================")
    print(f"Overall Accuracy:  {(y_true == y_pred).mean() * 100:.2f}%")
    print(f"Precision:         {precision * 100:.2f}% (When it predicts survival, it's right {precision*100:.1f}% of the time)")
    print(f"Recall:            {recall * 100:.2f}% (It successfully identified {recall*100:.1f}% of all actual survivals)")
    print(f"F1-Score:          {f1:.4f}")
    print(f"Brier Score:       {brier:.4f} (Probability Calibration Error - closer to 0 is better)")

    # 6. ERROR ISOLATION (THE BLIND SPOTS)
    print("\n=============================================")
    print(" HIGH-CONFIDENCE ERROR ISOLATION")
    print("=============================================")
    
    # Sort results by the absolute difference between prediction and reality
    for res in detailed_results:
        res['error_margin'] = abs(res['actual'] - res['predicted_prob'])
        
    detailed_results.sort(key=lambda x: x['error_margin'], reverse=True)
    
    # False Positives: Predicted > 95% survival, but player lost the ball (actual = 0)
    false_positives = [r for r in detailed_results if r['actual'] == 0 and r['predicted_prob'] > 0.90]
    
    # False Negatives: Predicted < 5% survival (Turnover imminent), but player survived (actual = 1)
    false_negatives = [r for r in detailed_results if r['actual'] == 1 and r['predicted_prob'] < 0.10]

    print("\n TOP 5 FALSE POSITIVES (Model thought they were safe, but they lost the ball)")
    print("These are usually technical errors by the player, or a pressing trap the model missed.")
    print("Frame\t\tTeam\tCarrier\t\tPredicted Survival")
    print("-" * 65)
    for res in false_positives[:5]:
        print(f"Frame {res['frame']:<8}\t{res['team']:<4}\tPlayer {res['carrier']:<6}\t{res['predicted_prob']*100:.1f}% safe")

    print("\n TOP 5 FALSE NEGATIVES (Model thought they were dead, but they escaped)")
    print("These are usually moments of individual brilliance, highly technical dribbling, or bad tackles.")
    print("Frame\t\tTeam\tCarrier\t\tPredicted Survival")
    print("-" * 65)
    for res in false_negatives[:5]:
        print(f"Frame {res['frame']:<8}\t{res['team']:<4}\tPlayer {res['carrier']:<6}\t{res['predicted_prob']*100:.1f}% safe")

if __name__ == "__main__":
    evaluate_model()