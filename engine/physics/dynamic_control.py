import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from .spatial_control import SpatialEngine

class DynamicSpatialEngine(SpatialEngine):
    """
    Calculates Dynamic Pitch Control by shifting Voronoi origins 
    forward along the player's momentum vector.
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, projection_time: float = 0.5):
        # Initialize the parent class to get the exact pitch boundaries
        super().__init__(pitch_length, pitch_width)
        
        # T: How many seconds into the future we project their momentum
        self.projection_time = projection_time

    def get_player_velocities(self, frame_data: pd.Series, coords: np.ndarray, team: str):
        """
        Matches spatial coordinates back to their velocity vectors.
        """
        projected_coords = []
        vx_list, vy_list = [], []
        
        start_idx = 1 if team == 'Home' else 15
        end_idx = 15 if team == 'Home' else 30

        for x, y in coords:
            found = False
            for i in range(start_idx, end_idx):
                col_x = f'{team}_Player_{i}_X'
                if col_x in frame_data and pd.notna(frame_data[col_x]) and abs(frame_data[col_x] - x) < 0.01:
                    vx = frame_data[f'{team}_Player_{i}_VX']
                    vy = frame_data[f'{team}_Player_{i}_VY']
                    
                    # The Physics Warp: Calculate the phantom coordinate
                    proj_x = x + (vx * self.projection_time)
                    proj_y = y + (vy * self.projection_time)
                    
                    projected_coords.append([proj_x, proj_y])
                    vx_list.append(vx)
                    vy_list.append(vy)
                    found = True
                    break
            
            # Fallback if velocity is completely zero or missing
            if not found:
                projected_coords.append([x, y])
                vx_list.append(0.0)
                vy_list.append(0.0)
                
        return np.array(projected_coords), vx_list, vy_list

    def calculate_dynamic_voronoi(self, frame_data: pd.Series):
        """
        Calculates geometrically clipped Voronoi regions based on FUTURE coordinates.
        """
        # 1. Use the locked parent method to get exactly 11 valid players
        h_coords, a_coords = self.extract_active_players(frame_data)
        
        # 2. Calculate their phantom coordinates based on momentum
        h_proj, h_vx, h_vy = self.get_player_velocities(frame_data, h_coords, 'Home')
        a_proj, a_vx, a_vy = self.get_player_velocities(frame_data, a_coords, 'Away')

        all_projected = np.vstack((h_proj, a_proj))

        # Dummy points for the mathematical boundaries
        boundary_points = np.array([[-1000, -1000], [1000, -1000], [1000, 1000], [-1000, 1000]])
        points_with_boundaries = np.vstack((all_projected, boundary_points))
        
        # 3. Calculate Voronoi on the PROJECTED points
        vor = Voronoi(points_with_boundaries)
        player_polygons = []
        
        # 4. The Cookie Cutter: Intersect with pitch boundaries
        for i in range(len(all_projected)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 not in region and len(region) > 0:
                polygon_coords = [vor.vertices[v] for v in region]
                poly = Polygon(polygon_coords)
                clipped_poly = poly.intersection(self.pitch_polygon) 
                player_polygons.append(clipped_poly)
            else:
                player_polygons.append(None)

        home_polygons = player_polygons[:len(h_coords)]
        away_polygons = player_polygons[len(h_coords):]

        # We return the REAL coordinates to draw the dots, but the WARPED polygons for space
        return home_polygons, away_polygons, h_coords, a_coords, h_vx, h_vy, a_vx, a_vy