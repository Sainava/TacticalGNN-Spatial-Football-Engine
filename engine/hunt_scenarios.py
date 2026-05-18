import pandas as pd

def hunt_via_events():
    print("Loading Metrica Event Data...")
    # Adjust the path to wherever your RawEventsData is located
    events_filepath = "../data/raw/Sample_Game_2/Sample_Game_2_RawEventsData.csv"
    
    try:
        events = pd.read_csv(events_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find Event Data at {events_filepath}")
        return

    print("\n--- EXACT TACTICAL EVENTS FOUND ---")

    # 1. HUNTING FOR GOALS
    # We look for any event of type 'SHOT' that resulted in a 'GOAL'
    shots = events[events['Type'] == 'SHOT']
    goals = shots[shots['Subtype'].astype(str).str.contains('-GOAL')]
    
    print("\nVERIFIED GOALS:")
    for index, goal in goals.iterrows():
        # We want to start the clip about 4 seconds (100 frames) BEFORE the shot is taken
        start_frame = int(goal['Start Frame']) - 100
        print(f"[{goal['Team']}] Goal by {goal['From']} -> Use Start Frame: {start_frame}")

    # 2. HUNTING FOR CRITICAL BALL LOSSES (High Press Turnovers)
    # We look for 'BALL LOST' events that happened deep in the defending team's half
    # Metrica coordinates are 0 to 1 for events. X < 0.3 means deep in their own third.
    turnovers = events[(events['Type'] == 'BALL LOST') & (events['Start X'] < 0.3)]
    
    print("\nCRITICAL DEEP TURNOVERS:")
    # Get the top 3 turnovers closest to the goal line
    worst_turnovers = turnovers.sort_values(by='Start X').head(3)
    for index, turnover in worst_turnovers.iterrows():
        start_frame = int(turnover['Start Frame']) - 50
        print(f"[{turnover['Team']}] Lost ball at X:{turnover['Start X']:.2f} -> Use Start Frame: {start_frame}")

    # 3. HUNTING FOR SET PIECES (Corners)
    # Great for testing high-density box scenarios
    corners = events[events['Subtype'].astype(str).str.contains('CORNER KICK')]
    
    print("\nCORNER KICKS:")
    first_few_corners = corners.head(2)
    for index, corner in first_few_corners.iterrows():
        # Start right as the kick is taken
        start_frame = int(corner['Start Frame']) - 25
        print(f"[{corner['Team']}] Corner Kick -> Use Start Frame: {start_frame}")

if __name__ == "__main__":
    hunt_via_events()