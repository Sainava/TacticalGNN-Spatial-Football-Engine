import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

class SpatialEngine:
    """
    Calculates spatial dominance using Bounded Voronoi Tessellations.
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        # Define the exact pitch boundary as a Shapely Polygon (The Cookie Cutter)
        self.pitch_polygon = Polygon([
            (-pitch_length/2, -pitch_width/2),
            (pitch_length/2, -pitch_width/2),
            (pitch_length/2, pitch_width/2),
            (-pitch_length/2, pitch_width/2)
        ])

    def extract_active_players(self, frame_data: pd.Series):
        """
        Extracts active players by enforcing pitch boundaries and deploying a 
        Technical Area Exclusion Zone to purge bench players.
        """
        home_x = [c for c in frame_data.index if c.startswith('Home_Player') and c.endswith('_X')]
        home_y = [c for c in frame_data.index if c.startswith('Home_Player') and c.endswith('_Y')]
        away_x = [c for c in frame_data.index if c.startswith('Away_Player') and c.endswith('_X')]
        away_y = [c for c in frame_data.index if c.startswith('Away_Player') and c.endswith('_Y')]

        h_coords = np.column_stack((frame_data[home_x].values, frame_data[home_y].values))
        a_coords = np.column_stack((frame_data[away_x].values, frame_data[away_y].values))

        # 1. Drop NaNs
        h_coords = h_coords[~np.isnan(h_coords).any(axis=1)]
        a_coords = a_coords[~np.isnan(a_coords).any(axis=1)]

        def filter_bench_players(coords):
            # If we already have 11 or fewer, we don't need to filter
            if len(coords) <= 11:
                return coords
            
            valid_players = []
            for p in coords:
                x, y = p[0], p[1]
                
                # THE TECHNICAL AREA TRAP: 
                # In Metrica Game 1, the bench is located at the bottom of the screen (Y ≈ -34)
                # near the halfway line (X between -20 and 20).
                # If a player is caught in this box, they are a substitute warming up.
                in_tech_area = (abs(x) < 20.0) and (y < -32.0)
                
                if not in_tech_area:
                    valid_players.append(p)
                    
            return np.array(valid_players)[:11] # Ensure we never return more than 11

        # Apply the filter
        h_coords = filter_bench_players(h_coords)
        a_coords = filter_bench_players(a_coords)

        return h_coords, a_coords
    
    def calculate_bounded_voronoi(self, frame_data: pd.Series):
        """
        Calculates Voronoi regions and geometrically clips them to the pitch grass.
        """
        h_coords, a_coords = self.extract_active_players(frame_data)
        all_players = np.vstack((h_coords, a_coords))

        # Dummy points far outside the pitch to close the mathematical regions
        boundary_points = np.array([
            [-1000, -1000], [1000, -1000], [1000, 1000], [-1000, 1000]
        ])
        points_with_boundaries = np.vstack((all_players, boundary_points))
        
        # Calculate raw mathematical Voronoi
        vor = Voronoi(points_with_boundaries)
        
        player_polygons = []
        
        # The Cookie Cutter: Intersect each player's infinite region with the pitch bounds
        for i in range(len(all_players)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 not in region and len(region) > 0:
                polygon_coords = [vor.vertices[v] for v in region]
                poly = Polygon(polygon_coords)
                clipped_poly = poly.intersection(self.pitch_polygon) # SHAPELY MAGIC HERE
                player_polygons.append(clipped_poly)
            else:
                player_polygons.append(None)

        # Split back into teams
        home_polygons = player_polygons[:len(h_coords)]
        away_polygons = player_polygons[len(h_coords):]

        return home_polygons, away_polygons, h_coords, a_coords