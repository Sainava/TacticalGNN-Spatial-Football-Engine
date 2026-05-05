# Tactical-Ghosting-Engine

```text
tactical-ghosting-engine/
│
├── .gitignore               # CRITICAL: We will put our data folders in here
├── README.md                # We will write the epic portfolio pitch here later
│
├── data/                    # The Storage (DO NOT PUSH TO GITHUB)
│   ├── raw/                 # Metrica CSVs and SoccerNet JSONs go here
│   ├── processed/           # Cleaned PyTorch tensors (.pt files)
│   └── ui_exports/          # The final JSON files the React app will read
│
├── engine/                  # The Brain (Python / PyTorch)
│   ├── requirements.txt
│   ├── config.py            # Store pitch dimensions, FPS rates, etc.
│   ├── parsers/
│   │   ├── metrica_parser.py   # Turns Metrica CSVs into standard format
│   │   └── soccernet_parser.py # Turns SoccerNet JSONs into standard format
│   ├── physics/
│   │   └── spatial_control.py  # The Voronoi and velocity heatmap math
│   ├── models/
│   │   └── st_gnn.py        # Your PyTorch Graph Neural Network
│   ├── train.py             # Script to train the model on Metrica data
│   └── export_to_ui.py      # Script to run inference and save to /ui_exports
│
└── dashboard/               # The Face (React / Vite)
    ├── package.json
    ├── public/
    │   └── assets/          # The actual SoccerNet .mp4 video files go here
    └── src/
        ├── App.jsx          # Main layout (Video top, Pitch bottom)
        ├── components/
        │   ├── VideoPlayer.jsx
        │   └── PitchMap.jsx # D3.js will draw the heatmaps here
        ├── hooks/
        │   └── useSyncFrame.js # Custom hook tying video time to JSON data
        └── index.css        # Sleek dark-mode styling
```