# TacticalGNN-Spatial-Football-Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-Vite-61DAFB.svg)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC.svg)](https://tailwindcss.com/)

> A deep learning and kinematics engine for spatial analytics in football, using a carrier-centric Graph Neural Network (GNN) to estimate short-term possession retention.

```text
tactical-ghosting-engine/
│
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ui_exports/
│
├── engine/                  # The Brain (Python / PyTorch)
│   ├── requirements.txt
│   ├── export_scenario.py
│   ├── hunt_scenarios.py
│   ├── visualize_pitch.py
│   ├── analytics/
│   │   ├── defensive_line.py
│   │   ├── animate_passing.py
│   │   ├── test_passing.py
│   │   ├── animate_defensive_line.py
│   │   └── passing_lanes.py
│   ├── models/
│   │   ├── gnn_architecture.py
│   │   ├── ccsn_builder.py
│   │   ├── ccsn_dataset.py
│   │   ├── inference_visualizer.py
│   │   ├── train_ccsnet.py
│   │   └── evaluate_ccsnet.py
│   ├── parsers/
│   │   └── metrica_parser.py
│   └── physics/
│       ├── kinematics.py
│       ├── spatial_control.py
│       ├── dynamic_control.py
│       ├── animate_dynamic.py
│       ├── test_dynamic.py
│       ├── test_voronoi.py
│       └── diagnose_frame.py
│
└── dashboard/               # The Face (React / Vite)
    ├── package.json
    ├── index.html
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── eslint.config.js
    ├── public/
    │   ├── assets/
    │   └── data/              # curated scenarios 
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── App.css
        ├── index.css
        └── components/
            ├── TacticalPitch.jsx
            ├── PressureChart.jsx
            └── (other UI components)
```

**[Live Interactive Dashboard](https://tactical-gnn-spatial-football-engin.vercel.app/)**

## System Overview

Standard optical tracking data in football provides $X/Y$ coordinates but does not directly encode contextual information such as dynamic pressure around the ball carrier. **TacticalGNN** bridges this gap by combining deterministic kinematic processing with Geometric Deep Learning.

This engine processes raw optical tracking data from Metrica Sports to calculate player kinematics, construct a carrier-centric spatial graph using distance, closing speed, and Time-To-Intercept (TTI), and use a Graph Neural Network (CCSNet architecture) to estimate the probability that the current carrier's team will retain possession three seconds into the future.

Event data is used separately to identify interesting match scenarios for visualization and export; it is not used to generate the GNN training labels.

![Dashboard Preview Placeholder](docs/assets/dashboard_preview.gif)
*Caption: The React telemetry dashboard replaying a 50-second, 18-pass build-up sequence using precomputed GNN and kinematic outputs.*

![Attacking Goal Sequence](docs/assets/Goal.gif)
*Caption: Final Third Conversion edge-case. The kinematic engine tracks player movement and dynamically recalculates Time-To-Intercept (TTI) as the attacking team penetrates the penalty box.*

## Key Analytical Features & Visualizations

The engine is split into specialized analytical modules, allowing for isolated testing and full pipeline integration.

### 1. Geometric Deep Learning (GNN) Layer

The core "brain" of the engine represents the pitch as a **carrier-centric directed ego-graph**.

* **Nodes & Edges:** Players are modeled as nodes, with the current ball carrier as the central node. Relevant surrounding players are connected to the carrier through dynamic directed edges.
* **Node Features:** Each player is represented using position, velocity, and team-relative information: `[X, Y, VX, VY, is_teammate]`.
* **Edge Features:** Relationships with the carrier include `[distance, TTI, closing_speed, is_teammate]`.
* **Possession Retention Inference:** The GNN estimates the probability that the carrier's team will still have possession three seconds later. TTI is used as an input edge feature rather than being directly predicted by the GNN.
* *(Backend Module: `engine/models/inference_visualizer.py`)*

![GNN Inference Placeholder](docs/assets/gnn_inference.gif)

### 2. Kinematic & Spatial Control Engine

Applies deterministic kinematic calculations to raw positional data to understand player movement and provide additional spatial context.

* **Dynamic Pitch Control:** Experimental modules extend standard Voronoi tessellations by projecting player positions forward using their current velocity before calculating spatial regions.
* **Velocity Vectors:** Derives X/Y velocity and speed from raw coordinate changes over time. The current implementation does not calculate acceleration.
* **Time-To-Intercept:** Estimates how quickly a surrounding player could reach the carrier using distance and closing speed.
* *(Backend Modules: `engine/physics/dynamic_control.py`, `engine/physics/kinematics.py`)*

![Pitch Control Placeholder](docs/assets/spatial_control.gif)

### 3. Tactical Shape Diagnostics

Additional experimental analytical modules are included for exploring tactical structure and passing behavior.

* **Defensive Line Tracking:** Experimental tactical-shape analysis.
* **Passing Networks:** Experimental visualization of ball progression and player relationships during possession.
* These modules are separate from the core GNN training and inference pipeline.

![Defensive Line Placeholder](docs/assets/defensive_line.gif)

## System Architecture

The project is structured as a monorepo, separating the heavy data processing/model training from the lightweight web visualizer.

```text
tactical-ghosting-engine/
├── engine/                  # The Brain (Python / PyTorch)
│   ├── analytics/           # Tactical shape and passing heuristics
│   ├── models/              # GNN architecture, CCSNet training & inference
│   ├── parsers/             # Ingestion pipelines for raw optical/event data
│   └── physics/             # Kinematics and spatial control calculations
└── dashboard/               # The Face (React / Vite / Tailwind)
    └── src/components/      # Interactive telemetry UI and canvas renderers

```

### The Engineering Pipeline

1. **Parser Layer:** Merges Home/Away tracking data, cleans the coordinates, and transforms them into a centered metric pitch representation.
2. **Kinematics Layer:** Computes player velocity and speed from positional changes over time.
3. **Graph Construction Layer:** Identifies the ball carrier and constructs a dynamic carrier-centric ego-graph using player kinematics, spatial relationships, and TTI.
4. **Inference Layer:** PyTorch GNN processes the graph and outputs a possession-retention probability.
5. **Export Layer:** Curated scenarios (Goals, Turnovers, Corners, and other selected sequences) are processed and serialized into lightweight JSON payloads.
6. **Presentation Layer:** A React dashboard renders the precomputed JSON outputs through an interactive SVG visualization.

## Tech Stack

* **Machine Learning & Data:** PyTorch, PyTorch Geometric, Pandas, NumPy, SciPy
* **Physics & Visualization (Python):** Matplotlib (for backend testing/diagnostics)
* **Frontend Web Application:** React.js, Vite, Tailwind CSS
* **Data Source:** [Metrica Sports Open Data](https://github.com/metrica-sports/sample-data)

##  Local Installation & Usage

### 1. The Python Engine (Backend)

Navigate to the `engine` directory to run data parsers, train models, or generate new scenarios.

```bash
cd engine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Example: Run the heuristic scenario hunter
python hunt_scenarios.py

# Example: Export curated scenarios to the React dashboard
python export_scenario.py

```

### 2. The React Dashboard (Frontend)

Navigate to the `dashboard` directory to run the visualizer.

```bash
cd dashboard
npm install
npm run dev

```

The dashboard will be available at `http://localhost:5173`.

## Acknowledgments

* Optical tracking and event datasets provided by [Metrica Sports](https://github.com/metrica-sports).

---

*Developed by Sainava Modak .*
