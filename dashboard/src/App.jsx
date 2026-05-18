import React, { useState, useEffect } from 'react';
import TacticalPitch from './components/TacticalPitch';
import PressureChart from './components/PressureChart';

// Centralized Scenario Registry - Final Curated Analytics Chapters
const SCENARIO_REGISTRY = [
  {
    id: 'masterclass_buildup',
    name: 'The Masterclass Build-Up',
    label: '1. Sustained Execution',
    file: '/data/scenario_masterclass.json',
    meta: 'Frame 3190 • 50.0s Window'
  },
  {
    id: 'midfield_possession',
    name: 'Sustained Midfield Possession',
    label: '2. Baseline Phase',
    file: '/data/scenario_midfield_possession.json',
    meta: 'Frame 65000 • 8.8s Window'
  },
  {
    id: 'attacking_goal',
    name: 'Attacking Phase & Finish',
    label: '3. Final Third Conversion',
    file: '/data/scenario_attacking_goal.json',
    meta: 'Frame 12102 • 8.8s Window'
  },
  {
    id: 'high_press',
    name: 'The High-Press Trap',
    label: '4. Defensive Phase',
    file: '/data/scenario_high_press.json',
    meta: 'Frame 42200 • 7.2s Window'
  },
  {
    id: 'corner_kick',
    name: 'Set Piece Dynamics',
    label: '5. Maximum Density',
    file: '/data/scenario_corner_kick.json',
    meta: 'Frame 10438 • 8.0s Window'
  },
  {
    id: 'tight_spaces',
    name: 'Possession in Tight Spaces',
    label: '6. Constriction Phase',
    file: '/data/scenario_tight_spaces.json',
    meta: 'Frame 93200 • 8.0s Window'
  },
  {
    id: 'penalty_kick',
    name: 'The Penalty Kick',
    label: '7. Dead Ball Edge Case',
    file: '/data/scenario_penalty_kick.json',
    meta: 'Frame 115000 • 8.8s Window'
  }
];

function App() {
  const [data, setData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Track active file from registry
  const [activeFile, setActiveFile] = useState(SCENARIO_REGISTRY[0].file);
  const [viewMode, setViewMode] = useState('gnn'); // NEW STATE: 'raw', 'kinematics', or 'gnn'

  // Fetch data on scenario switch
  useEffect(() => {
    setIsPlaying(false);
    setData(null);
    
    fetch(activeFile)
      .then((res) => res.json())
      .then((jsonData) => {
        setData(jsonData);
        setCurrentFrame(0);
      })
      .catch((err) => console.error("Pipeline read failure:", err));
  }, [activeFile]);

  // Global Animation Clock (25 FPS)
  useEffect(() => {
    if (isPlaying && data) {
      const interval = setInterval(() => {
        setCurrentFrame((prev) => {
          if (prev >= data.metadata.num_frames - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 40);
      return () => clearInterval(interval);
    }
  }, [isPlaying, data]);

  // Compute telemetry telemetry values for active frame
  const frameData = data?.frames[currentFrame];
  const carrierText = frameData?.carrier?.id || "None / Loose Ball";
  const probText = frameData?.prob !== undefined ? (frameData.prob * 100).toFixed(1) + "%" : "--";
  
  let primaryThreat = "None";
  let minTti = "--";
  if (frameData?.edges && frameData.edges.length > 0) {
    const sortedEdges = [...frameData.edges].sort((a, b) => a.tti - b.tti);
    primaryThreat = `Defender (ID: ${sortedEdges[0].tti < 1.5 ? 'CRITICAL' : 'CLOSE' })`;
    minTti = sortedEdges[0].tti.toFixed(2);
  }

  return (
      <div className="h-screen w-screen flex font-sans bg-slate-950">
        
        {/* LEFT SIDEBAR: DYNAMIC SCENARIO SELECTOR */}
        <div className="w-64 bg-slate-800/40 border-r border-slate-700 flex flex-col backdrop-blur-md z-10">
          <div className="p-6 border-b border-slate-700/50">
            <h1 className="text-lg font-bold tracking-wider text-white">TACTICAL<span className="text-emerald-400">GNN</span></h1>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Spatial Engine Suite</p>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-2 overflow-y-auto">
            <p className="text-xs text-slate-500 uppercase tracking-widest mb-2 font-semibold">Analytical Chapters</p>
            
            {SCENARIO_REGISTRY.map((scenario) => {
              const isActive = activeFile === scenario.file;
              return (
                <button
                  key={scenario.id}
                  onClick={() => setActiveFile(scenario.file)}
                  className={`text-left px-4 py-3 rounded-lg border-l-4 transition-all ${
                    isActive 
                      ? 'bg-slate-700/50 border-emerald-400 shadow-lg' 
                      : 'bg-transparent border-transparent hover:bg-slate-800/60'
                  }`}
                >
                  <span className="block text-[10px] font-bold text-emerald-400/80 uppercase tracking-wider mb-0.5">
                    {scenario.label}
                  </span>
                  <span className={`block text-sm font-semibold ${isActive ? 'text-white' : 'text-slate-300'}`}>
                    {scenario.name}
                  </span>
                  <span className="block text-xs text-slate-500 mt-1 font-mono">
                    {scenario.meta}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* MAIN LAYOUT MATRICES */}
        <div className="flex-1 flex flex-col p-6 gap-6 relative min-w-0">
          
          {/* NEW: MINIMALIST TOP NAV BAR CONTAINER */}
          <div className="w-full flex items-center justify-between border-b border-slate-800/60 pb-4 shrink-0">
            <div>
              <h1 className="text-xl font-bold text-white tracking-wide">
                TacticalGNN: <span className="text-emerald-400 font-light">Spatial Football Engine</span>
              </h1>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Deep Learning Kinematic Threat Matrix • Metrica Analytics Suite
              </p>
            </div>

            {/* GLOWING GITHUB ACTION TARGET */}
            <a 
              href="https://github.com/Sainava/TacticalGNN-Spatial-Football-Engine" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-emerald-500/50 bg-emerald-900/20 hover:bg-emerald-800/40 text-emerald-400 hover:text-emerald-300 font-mono text-xs transition-all duration-300 tracking-wide shadow-[0_0_15px_rgba(52,211,153,0.3)] hover:shadow-[0_0_25px_rgba(52,211,153,0.6)]"
            >
              <svg className="w-4 h-4 fill-current drop-shadow-[0_0_5px_rgba(52,211,153,0.8)]" viewBox="0 0 24 24" aria-hidden="true">
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.483 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.008.069-.008 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.138 20.161 22 16.416 22 12c0-5.523-4.477-10-10-10z" />
              </svg>
              <span>SOURCE_CODE</span>
            </a>
          </div>

          {/* DATA CONTAINER GRIDS */}
          <div className="flex-1 flex gap-6 min-h-0">
            
            {/* THE CANVAS */}
            <div className="flex-1 bg-slate-800/30 rounded-2xl border border-slate-700/50 shadow-2xl flex items-center justify-center relative overflow-hidden backdrop-blur-sm">
              <TacticalPitch 
                data={data} 
                currentFrame={currentFrame} 
                isPlaying={isPlaying} 
                setIsPlaying={setIsPlaying} 
                setCurrentFrame={setCurrentFrame} 
                viewMode={viewMode}
              />
            </div>

            {/* RIGHT COLUMN: PIPELINE CONTROLS & TELEMETRY */}
            <div className="w-80 flex flex-col gap-6 shrink-0">
              
              {/* THE PIPELINE STORY TOGGLES */}
              <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-4 backdrop-blur-sm">
                <h2 className="text-[10px] uppercase tracking-widest text-emerald-400/80 font-bold mb-3">Engineering Pipeline</h2>
                <div className="flex flex-col gap-2">
                  <button 
                    onClick={() => setViewMode('raw')}
                    className={`text-left px-3 py-2 rounded text-sm transition-colors border-l-2 ${viewMode === 'raw' ? 'bg-slate-700 border-white text-white' : 'border-transparent text-slate-400 hover:bg-slate-700/50'}`}
                  >
                    1. Raw Optical Data
                  </button>
                  <button 
                    onClick={() => setViewMode('kinematics')}
                    className={`text-left px-3 py-2 rounded text-sm transition-colors border-l-2 ${viewMode === 'kinematics' ? 'bg-slate-700 border-white text-white' : 'border-transparent text-slate-400 hover:bg-slate-700/50'}`}
                  >
                    2. Kinematics Engine
                  </button>
                  <button 
                    onClick={() => setViewMode('gnn')}
                    className={`text-left px-3 py-2 rounded text-sm transition-colors border-l-2 ${viewMode === 'gnn' ? 'bg-slate-700 border-emerald-400 text-emerald-400 font-semibold' : 'border-transparent text-slate-400 hover:bg-slate-700/50'}`}
                  >
                    3. GNN Inference (Live)
                  </button>
                </div>
              </div>

              {/* DYNAMIC TELEMETRY PANEL */}
              <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-6 flex-1 flex flex-col backdrop-blur-sm">
                <h2 className="text-sm uppercase tracking-widest text-slate-400 font-semibold border-b border-slate-700 pb-2 mb-4">Live Telemetry</h2>
                <div className="flex flex-col gap-5">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider">Active Ball Carrier</p>
                    <p className="text-lg text-white font-mono mt-0.5">{carrierText}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider">Interception Vector</p>
                    <p className="text-lg text-rose-400 font-mono mt-0.5">{primaryThreat}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider">Time-To-Intercept (TTI)</p>
                    <p className="text-3xl text-white font-mono font-light mt-0.5">{minTti}<span className="text-lg text-slate-400">s</span></p>
                  </div>
                  <div className="pt-4 border-t border-slate-800">
                    <p className="text-xs text-slate-500 uppercase tracking-wider">GNN Core Evaluation</p>
                    <p className={`text-3xl font-mono font-bold mt-1 ${
                      frameData?.prob > 0.75 ? 'text-emerald-400' : frameData?.prob > 0.35 ? 'text-yellow-400' : 'text-rose-400'
                    }`}>
                      {probText}
                    </p>
                  </div>
                </div>
              </div>
            </div>

          </div>

          {/* TIME SERIES HEARTBEAT */}
          <div className="h-48 bg-slate-800/30 rounded-2xl border border-slate-700/50 p-4 flex flex-col backdrop-blur-sm overflow-hidden shrink-0">
            <div className="flex justify-between items-center mb-2 px-2 z-10 relative">
              <h2 className="text-sm uppercase tracking-widest text-slate-400 font-semibold">Tactical Pressure Heartbeat</h2>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded">Live Graph Inference</span>
            </div>
            <div className="flex-1 w-full rounded-lg overflow-hidden">
              <PressureChart data={data} currentFrame={currentFrame} />
            </div>
          </div>

        </div>
      </div>
    );
  }

export default App;
