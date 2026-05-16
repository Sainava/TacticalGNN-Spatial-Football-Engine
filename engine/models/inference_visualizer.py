import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle, Circle
import sys
import os

# Import your pipeline components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from models.gnn_architecture import CCSNet
from models.ccsn_builder import CCSNBuilder

class TacticalVisualizer:
    def __init__(self, model_path, device_type='mps'):
        self.device = torch.device(device_type if torch.backends.mps.is_available() else "cpu")
        print(f"Visualizer Engine locked and loaded on: {self.device}")
        
        self.model = CCSNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.builder = CCSNBuilder() 
        self.max_tti = 3.5
        self.max_dist = 15.0

    def get_survival_probability(self, players, carrier, play_direction):
        flip = -1 if play_direction == "Left" else 1
        nodes, edge_sources, edge_targets, edge_attr = [], [], [], []
        
        carrier_pos = carrier['pos'] * flip
        carrier_vel = carrier['vel'] * flip
        nodes.append([carrier_pos[0], carrier_pos[1], carrier_vel[0], carrier_vel[1], 1.0])
        
        visual_edges = []
        for i, p in enumerate(players):
            if p['id'] == carrier['id']: continue
            
            is_teammate = 1.0 if p['team'] == carrier['team'] else 0.0
            norm_p_pos = p['pos'] * flip
            norm_p_vel = p['vel'] * flip
            
            vec = carrier_pos - norm_p_pos
            dist = np.linalg.norm(vec)
            if dist == 0: dist = 1e-9
            
            closing_speed = np.dot(norm_p_vel, vec / dist)
            tti = dist / closing_speed if closing_speed > 0 else 10.0
            
            if tti <= self.max_tti or dist <= self.max_dist:
                nodes.append([norm_p_pos[0], norm_p_pos[1], norm_p_vel[0], norm_p_vel[1], is_teammate])
                edge_sources.append(len(nodes) - 1)
                edge_targets.append(0)
                edge_attr.append([dist, min(tti, 10.0), closing_speed, is_teammate])
                if not is_teammate:
                    visual_edges.append({'pos': p['pos'], 'tti': tti})

        if len(edge_sources) == 0: return 0.99, []

        x = torch.tensor(nodes, dtype=torch.float).to(self.device)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long).to(self.device)
        attr = torch.tensor(edge_attr, dtype=torch.float).to(self.device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x, edge_index, attr, batch)
            prob = torch.sigmoid(logits).item()
            
        return prob, visual_edges

    def precompute_clip(self, df, start_frame, num_frames):
        print(f"Pre-computing {num_frames} frames to eliminate lag...")
        results = {}
        for i in range(num_frames):
            frame_idx = start_frame + i
            if frame_idx >= len(df): break
            frame = df.iloc[frame_idx]
            
            if np.isnan(frame['Home_Ball_X']):
                results[frame_idx] = {'status': 'dead_ball'}
                continue

            players, ball = self.builder.extract_nodes_from_frame(frame)
            if not players: continue
            
            carrier = self.builder.identify_carrier(players, ball)
            direction = self.builder.get_play_direction(frame, carrier['team'])
            
            if np.linalg.norm(carrier['pos'] - ball) > 2.0:
                results[frame_idx] = {'status': 'loose_ball', 'players': players, 'ball': ball}
            else:
                prob, edges = self.get_survival_probability(players, carrier, direction)
                results[frame_idx] = {'status': 'possession', 'players': players, 'ball': ball, 'carrier': carrier, 'prob': prob, 'edges': edges}
        return results

    def animate_tactical_state(self, df, start_frame=88888, num_frames=300):
        fps = 25.0 
        frame_data = self.precompute_clip(df, start_frame, num_frames)
        
        # ==========================================
        # SETUP FIGURE AND PITCH
        # ==========================================
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a1a') 
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, hspace=0.3)
        
        ax_pitch = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax_pitch.set_facecolor('#2e7d43') # THE METRICA GREEN
        
        ax_pitch.set_xlim(-55, 55)
        ax_pitch.set_ylim(-35, 35)
        
        # Turn axes ON and make spines white
        ax_pitch.axis('on') 
        ax_pitch.tick_params(colors='white', labelsize=10)
        for spine in ax_pitch.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1.5)
        
        # Draw Pitch Lines
        ax_pitch.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, color='white', linewidth=1.5))
        ax_pitch.add_patch(Circle((0,0), 9.15, fill=False, color='white', linewidth=1.5))
        ax_pitch.axvline(0, color='white', linewidth=1.5)
        ax_pitch.add_patch(Rectangle((-52.5, -20.15), 16.5, 40.3, fill=False, color='white', linewidth=1.5))
        ax_pitch.add_patch(Rectangle((36, -20.15), 16.5, 40.3, fill=False, color='white', linewidth=1.5))
        
       # ==========================================
        # SETUP HEARTBEAT CHART
        # ==========================================
        ax_chart = plt.subplot2grid((3, 1), (2, 0))
        ax_chart.set_facecolor('#111111') # Ultra dark background
        ax_chart.set_xlim(0, num_frames / fps)
        ax_chart.set_ylim(0, 1.05)
        
        # 1. Clean, Solid Dark-Mode Zones (No muddy alpha blending)
        ax_chart.axhspan(0.75, 1.05, color='#0f291e', alpha=1.0) # Deep Forest Green
        ax_chart.axhspan(0.35, 0.75, color='#2e2713', alpha=1.0) # Deep Bronze/Yellow
        ax_chart.axhspan(0.00, 0.35, color='#2e1313', alpha=1.0) # Deep Crimson
        
        # 2. Explicit Zone Watermarks
        ax_chart.text(0.1, 0.90, 'HIGH SAFETY (>75%)', color='#2ecc71', fontsize=11, fontweight='bold', alpha=0.6)
        ax_chart.text(0.1, 0.55, 'MODERATE RISK (35-75%)', color='#f1c40f', fontsize=11, fontweight='bold', alpha=0.6)
        ax_chart.text(0.1, 0.15, 'CRITICAL DANGER (<35%)', color='#e74c3c', fontsize=11, fontweight='bold', alpha=0.6)
        
        # Styling the chart axes
        ax_chart.spines['bottom'].set_color('#444444')
        ax_chart.spines['left'].set_color('#444444')
        ax_chart.spines['top'].set_visible(False)
        ax_chart.spines['right'].set_visible(False)
        ax_chart.tick_params(colors='#aaaaaa', labelsize=10)
        ax_chart.set_xlabel('Time (Seconds)', color='#aaaaaa', fontsize=12)
        ax_chart.set_ylabel('Retention Probability', color='#aaaaaa', fontsize=12)
        ax_chart.set_title('Tactical Pressure Heartbeat', color='white', fontsize=14, pad=10)
        
        # ==========================================
        # INITIALIZE SCATTER OBJECTS
        # ==========================================
        scat_home = ax_pitch.scatter([], [], c='#ff0000', s=150, edgecolors='white', linewidth=2, zorder=4)
        scat_away = ax_pitch.scatter([], [], c='#0000ff', s=150, edgecolors='white', linewidth=2, zorder=4)
        scat_ball = ax_pitch.scatter([], [], c='black', s=80, edgecolors='white', linewidth=2, zorder=6)
        
        halo = Ellipse((0,0), 12, 12, color='green', alpha=0.3, zorder=2)
        ax_pitch.add_patch(halo)
        
        text_info = ax_pitch.text(-51, 30, "", color='white', fontsize=14, fontweight='bold', zorder=6)
        threat_lines = [ax_pitch.plot([], [], color='red', linewidth=2, linestyle='--', alpha=0)[0] for _ in range(5)]
        
        time_x, prob_y = [], []
        # Make the main line slightly thinner and bright white
        line_chart, = ax_chart.plot([], [], color='#ffffff', linewidth=2.5, zorder=3)
        current_time_marker = ax_chart.axvline(0, color='white', linestyle=':', alpha=0.4, linewidth=1)
        
        # Add a dynamic tracking dot at the leading edge
        current_dot, = ax_chart.plot([], [], 'o', color='white', markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=2)

        # ==========================================
        # FAST UPDATE LOOP (NO ax.clear() ALLOWED)
        # ==========================================
        def update(i):
            frame_idx = start_frame + i
            if frame_idx not in frame_data: return
            
            data = frame_data[frame_idx]
            current_time_sec = i / fps
            
            if data['status'] == 'dead_ball':
                text_info.set_text(f"Frame {frame_idx}: Dead Ball")
                halo.set_alpha(0)
                return
                
            home_coords = [p['pos'] for p in data['players'] if p['team'] == 'Home']
            away_coords = [p['pos'] for p in data['players'] if p['team'] == 'Away']
            
            if home_coords: scat_home.set_offsets(home_coords)
            if away_coords: scat_away.set_offsets(away_coords)
            scat_ball.set_offsets([data['ball'][0], data['ball'][1]])
            
            if data['status'] == 'loose_ball':
                text_info.set_text(f"Frame {frame_idx}: Pass in Air / Loose Ball")
                halo.set_alpha(0)
                for line in threat_lines: line.set_alpha(0)
                
                if len(prob_y) > 0:
                    time_x.append(current_time_sec)
                    prob_y.append(prob_y[-1]) 
                    line_chart.set_data(time_x, prob_y)
                current_time_marker.set_xdata([current_time_sec])
                return
                
            carrier = data['carrier']
            prob = data['prob']
            edges = data['edges']
            
            halo.set_center(carrier['pos'])
            if prob > 0.75: halo_color = '#2ecc71'
            elif prob > 0.35: halo_color = '#f1c40f'
            else: halo_color = '#e74c3c'
            
            halo.set_color(halo_color)
            halo.set_alpha(0.4)
            
            for j, line in enumerate(threat_lines):
                if j < len(edges):
                    target = edges[j]
                    line.set_data([carrier['pos'][0], target['pos'][0]], [carrier['pos'][1], target['pos'][1]])
                    if target['tti'] < 1.5:
                        line.set_color('#e74c3c'); line.set_linewidth(2.5); line.set_alpha(0.8)
                    else:
                        line.set_color('yellow'); line.set_linewidth(1.5); line.set_alpha(0.5)
                else:
                    line.set_alpha(0)
            
            text_info.set_text(f"Frame {frame_idx} | Carrier: {carrier['id']} | Survival: {prob*100:.1f}%")
            
           # Update Chart
            time_x.append(current_time_sec)
            prob_y.append(prob)
            line_chart.set_data(time_x, prob_y)
            current_time_marker.set_xdata([current_time_sec])
            
            # 3. Update the Live Tracker Dot (Matches the Halo color!)
            current_dot.set_data([current_time_sec], [prob])
            current_dot.set_color(halo_color)

        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=40, repeat=False)
        plt.show()

if __name__ == "__main__":
    
    viz = TacticalVisualizer("../../models/ccsnet_best_model.pth")
    parser = MetricaParser()
    clean_df = parser.merge_and_clean("../../data/raw/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv", 
                                      "../../data/raw/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv")
    momentum_df = KinematicsEngine().add_kinematics(clean_df)
    
    viz.animate_tactical_state(momentum_df, start_frame=88888, num_frames=200)