import React from 'react';

const TacticalPitch = ({ data, currentFrame, isPlaying, setIsPlaying, setCurrentFrame, viewMode = 'gnn' }) => {
  if (!data) return <div className="text-emerald-400 font-mono animate-pulse">Loading Tactical Matrix...</div>;

  const frameData = data.frames[currentFrame];
  if (!frameData) return null;

  let haloColor = 'rgba(46, 204, 113, 0.4)';
  if (frameData.prob !== undefined) {
    if (frameData.prob <= 0.35) haloColor = 'rgba(231, 76, 60, 0.5)';
    else if (frameData.prob <= 0.75) haloColor = 'rgba(241, 196, 15, 0.4)';
  }

  return (
    <div className="w-full h-full flex flex-col relative">
      <svg viewBox="-55 -35 110 70" className="w-full h-full bg-[#2e7d43] rounded-t-xl" preserveAspectRatio="xMidYMid meet">
        {/* PITCH MARKINGS */}
        <g stroke="rgba(255,255,255,0.7)" strokeWidth="0.3" fill="none">
          <rect x="-52.5" y="-34" width="105" height="68" />
          <circle cx="0" cy="0" r="9.15" />
          <line x1="0" y1="-34" x2="0" y2="34" />
          <rect x="-52.5" y="-20.15" width="16.5" height="40.3" />
          <rect x="36" y="-20.15" width="16.5" height="40.3" />
        </g>

        {/* LAYER 3: GNN INFERENCE (Threat Lines & Halo) */}
        {viewMode === 'gnn' && frameData.status === 'possession' && (
          <>
            {frameData.edges?.map((edge, i) => (
              <line key={`edge-${i}`} x1={frameData.carrier.pos[0]} y1={frameData.carrier.pos[1]} x2={edge.pos[0]} y2={edge.pos[1]}
                stroke={edge.tti < 1.5 ? '#e74c3c' : '#f1c40f'} strokeWidth={edge.tti < 1.5 ? "0.6" : "0.3"} strokeDasharray="1, 1" opacity={edge.tti < 1.5 ? 0.9 : 0.5} />
            ))}
            <circle cx={frameData.carrier.pos[0]} cy={frameData.carrier.pos[1]} r="8" fill={haloColor} />
          </>
        )}

        {/* LAYER 2: KINEMATICS (Velocity Vectors) */}
        {viewMode === 'kinematics' && frameData.players?.map((p) => {
          // If velocity data exists, draw a momentum arrow indicating where they will be in 1 second
          if (p.vel) {
            return (
              <line 
                key={`vel-${p.id}`} 
                x1={p.pos[0]} y1={p.pos[1]} 
                x2={p.pos[0] + (p.vel[0] * 1.0)} y2={p.pos[1] + (p.vel[1] * 1.0)} 
                stroke="rgba(255,255,255,0.8)" strokeWidth="0.4" 
              />
            );
          }
          return null;
        })}

        {/* LAYER 1: RAW DATA (The Players) - Always visible */}
        {frameData.players?.map((p) => (
          <circle key={p.id} cx={p.pos[0]} cy={p.pos[1]} r="1.2" fill={p.team === 'Home' ? '#ff0000' : '#0000ff'} stroke="white" strokeWidth="0.4" />
        ))}

        {/* THE BALL */}
        {frameData.ball && (
          <circle cx={frameData.ball[0]} cy={frameData.ball[1]} r="0.8" fill="black" stroke="white" strokeWidth="0.3" />
        )}
      </svg>

      {/* SCRUBBER CONTROLS */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-900/80 backdrop-blur-md border border-slate-700 px-4 py-2 rounded-full flex items-center gap-4">
        <button onClick={() => setIsPlaying(!isPlaying)} className="text-emerald-400 hover:text-emerald-300 font-bold w-12">
          {isPlaying ? 'PAUSE' : 'PLAY'}
        </button>
        <input type="range" min="0" max={data.metadata.num_frames - 1} value={currentFrame}
          onChange={(e) => { setIsPlaying(false); setCurrentFrame(parseInt(e.target.value)); }}
          className="w-48 accent-emerald-500 cursor-pointer" />
        <span className="text-slate-400 font-mono text-xs w-12">{frameData.time_sec.toFixed(1)}s</span>
      </div>
    </div>
  );
};

export default TacticalPitch;