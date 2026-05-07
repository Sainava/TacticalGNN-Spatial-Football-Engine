import numpy as np

class DefensiveLineEngine:
    """
    Tracks the physical shape of the defending back four and their 
    isolation distance to the primary target attacker.
    """
    
    def get_back_four_chain(self, def_coords, defending_side):
        """Finds the 4 deepest outfield players based on which goal they are defending."""
        if defending_side == 'left':
            # Defending goal at X = -52.5. Deepest players have the most negative X.
            sorted_by_depth = def_coords[np.argsort(def_coords[:, 0])]
        else:
            # Defending goal at X = 52.5. Deepest players have the most positive X.
            sorted_by_depth = def_coords[np.argsort(-def_coords[:, 0])]
            
        # Exclude the Goalkeeper (index 0), grab the next 4
        back_four = sorted_by_depth[1:5]
        
        # Sort those 4 players by their Y coordinate (bottom flank to top flank)
        back_four_chain = back_four[np.argsort(back_four[:, 1])]
        return back_four_chain

    def get_target_attacker(self, att_coords, attacking_side):
        """Finds the furthest forward opposition player based on attack direction."""
        if attacking_side == 'right':
            # Attacking towards X = 52.5
            target_idx = np.argmax(att_coords[:, 0])
        else:
            # Attacking towards X = -52.5
            target_idx = np.argmin(att_coords[:, 0])
            
        return att_coords[target_idx]
        
    def get_nearest_marker(self, attacker_coord, back_four_coords):
        """Finds the specific defender closest to the target attacker."""
        distances = np.linalg.norm(back_four_coords - attacker_coord, axis=1)
        nearest_idx = np.argmin(distances)
        
        return back_four_coords[nearest_idx], distances[nearest_idx]