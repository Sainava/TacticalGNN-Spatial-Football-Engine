# Tactical-Ghosting-Engine

```text
tactical-ghosting-engine/
│
├── .gitignore               # TO Ignore Files 
├── README.md                # Readme
│
├── data/                    # Storage
│   ├── raw/                 # Metrica CSVs and SoccerNet JSONs
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
│   │   └── st_gnn.py        # PyTorch Graph Neural Network
│   ├── train.py             # Script to train the model on Metrica data
│   └── export_to_ui.py      # Script to run inference and save to /ui_exports
│
└── dashboard/               # The Face (React / Vite)
    ├── package.json
    ├── public/
    │   └── assets/          #SoccerNet .mp4 video files
    └── src/
        ├── App.jsx          # Main layout (Video top, Pitch bottom)
        ├── components/
        │   ├── VideoPlayer.jsx
        │   └── PitchMap.jsx # D3.js will draw the heatmaps here
        ├── hooks/
        │   └── useSyncFrame.js # Custom hook tying video time to JSON data
        └── index.css        # Sleek dark-mode styling
```