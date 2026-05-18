import React from 'react';

const PressureChart = ({ data, currentFrame }) => {
  if (!data || !data.frames) return <div className="text-slate-500 animate-pulse">Waiting for telemetry...</div>;

  const totalFrames = data.metadata.num_frames;
  
  // Create the SVG path using a 1000-unit wide canvas instead of 100
  let pathD = "";
  data.frames.forEach((f, i) => {
    // Map X to 0-1000 to match the wide aspect ratio
    const x = (i / (totalFrames - 1)) * 1000;
    const y = f.prob !== undefined ? (1 - f.prob) * 100 : 100; // Drop to bottom if loose ball
    
    if (i === 0) pathD += `M ${x} ${y} `;
    else pathD += `L ${x} ${y} `;
  });

  // Calculate live dot position on the 1000-unit scale
  const currentF = data.frames[currentFrame];
  const dotX = (currentFrame / (totalFrames - 1)) * 1000;
  const dotY = currentF && currentF.prob !== undefined ? (1 - currentF.prob) * 100 : 100;

  // Dot color matching the pitch halo
  let dotColor = '#2ecc71'; // Green
  if (currentF && currentF.prob !== undefined) {
    if (currentF.prob <= 0.35) dotColor = '#e74c3c';
    else if (currentF.prob <= 0.75) dotColor = '#f1c40f';
  }

  return (
    <div className="w-full h-full relative">
      {/* Updated viewBox to 1000x100 for a natural wide aspect ratio */}
      <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 1000 100">
        
        {/* Interpretability Zones (Updated width to 1000) */}
        <rect x="0" y="0" width="1000" height="25" fill="#0f291e" opacity="0.8" /> 
        <rect x="0" y="25" width="1000" height="40" fill="#2e2713" opacity="0.8" /> 
        <rect x="0" y="65" width="1000" height="35" fill="#2e1313" opacity="0.8" /> 
        
        {/* Probability Line with vectorEffect to prevent blocky stretching */}
        <path 
          d={pathD} 
          fill="none" 
          stroke="white" 
          strokeWidth="2" 
          strokeLinejoin="round" 
          vectorEffect="non-scaling-stroke" 
        />
        
        {/* Current Time Scrubber Line */}
        <line 
          x1={dotX} y1="0" x2={dotX} y2="100" 
          stroke="rgba(255,255,255,0.4)" 
          strokeWidth="1" 
          strokeDasharray="4,4" 
          vectorEffect="non-scaling-stroke" 
        />
        
        {/* Live Tracker Dot */}
        <circle 
          cx={dotX} cy={dotY} r="4" 
          fill={dotColor} 
          stroke="white" 
          strokeWidth="1.5" 
          vectorEffect="non-scaling-stroke" 
        />
      </svg>
      
      {/* Zone Watermarks (Added pointer-events-none so they don't block hovering) */}
      <div className="absolute top-2 left-2 text-[10px] font-bold text-[#2ecc71]/50 tracking-wider pointer-events-none">HIGH SAFETY (&gt;75%)</div>
      <div className="absolute top-[35%] left-2 text-[10px] font-bold text-[#f1c40f]/50 tracking-wider pointer-events-none">MODERATE RISK</div>
      <div className="absolute bottom-4 left-2 text-[10px] font-bold text-[#e74c3c]/50 tracking-wider pointer-events-none">CRITICAL DANGER (&lt;35%)</div>
    </div>
  );
};

export default PressureChart;