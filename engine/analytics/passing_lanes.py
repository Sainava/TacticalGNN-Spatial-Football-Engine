import numpy as np
from shapely.geometry import LineString

class PassingViabilityEngine:
    """
    Calculates open and blocked passing lanes using geometric ray casting.
    A pass is blocked if the line between the passer and receiver heavily 
    intersects an opponent's spatial control polygon.
    """
    
    def evaluate_passing_lanes(self, passer_coord, teammate_coords, opponent_polygons):
        """
        Evaluates all passing options for the ball carrier.
        
        Returns:
            open_passes: list of (x, y) coordinates for open teammates
            blocked_passes: list of (x, y) coordinates for blocked teammates
        """
        open_passes = []
        blocked_passes = []
        
        # Filter out empty polygons
        valid_opp_polys = [p for p in opponent_polygons if p and not p.is_empty]
        
        for tm_coord in teammate_coords:
            # Skip drawing a pass to yourself
            if np.array_equal(passer_coord, tm_coord):
                continue
                
            pass_line = LineString([passer_coord, tm_coord])
            is_blocked = False
            
            for opp_poly in valid_opp_polys:
                # Calculate the intersection between the pass and the defender's zone
                intersection = pass_line.intersection(opp_poly)
                
                # If the pass line cuts through more than 0.1 meters of an enemy polygon,
                # it is structurally blocked. (0.1m threshold ignores tiny edge-touches)
                if intersection.length > 0.1: 
                    is_blocked = True
                    break
                    
            if is_blocked:
                blocked_passes.append(tm_coord)
            else:
                open_passes.append(tm_coord)
                
        return open_passes, blocked_passes
    
    def find_ball_carrier(self, ball_coord, h_coords, a_coords):
        """
        Calculates euclidean distance to find which player has the ball.
        """
        h_dists = np.linalg.norm(h_coords - ball_coord, axis=1)
        a_dists = np.linalg.norm(a_coords - ball_coord, axis=1)
        
        min_h, min_a = np.min(h_dists), np.min(a_dists)
        
        if min_h < min_a:
            idx = np.argmin(h_dists)
            return 'Home', h_coords[idx]
        else:
            idx = np.argmin(a_dists)
            return 'Away', a_coords[idx]