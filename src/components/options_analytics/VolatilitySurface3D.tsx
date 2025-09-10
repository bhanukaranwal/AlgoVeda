/*!
 * Interactive 3D Volatility Surface Visualization
 * WebGL-powered real-time volatility surface rendering
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Box, Paper, Typography, Slider, FormControlLabel, Switch } from '@mui/material';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';

interface VolatilitySurfaceProps {
  volatilityData: {
    strikes: number[];
    expirations: number[];
    impliedVols: number[][];
  };
  underlyingPrice: number;
  riskFreeRate: number;
  showGrid?: boolean;
  colorScheme?: 'thermal' | 'rainbow' | 'grayscale';
}

// WebGL Volatility Surface Component
const VolatilitySurfaceMesh: React.FC<{
  data: number[][];
  strikes: number[];
  expirations: number[];
  colorScheme: string;
}> = ({ data, strikes, expirations, colorScheme }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Generate surface geometry
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(10, 10, strikes.length - 1, expirations.length - 1);
    const vertices = geo.attributes.position.array;
    const colors = [];
    
    // Map volatility data to 3D surface
    for (let i = 0; i < expirations.length; i++) {
      for (let j = 0; j < strikes.length; j++) {
        const index = i * strikes.length + j;
        const vol = data[i][j];
        
        // Set Z coordinate based on volatility
        vertices[index * 3 + 2] = vol * 5; // Scale volatility
        
        // Color based on volatility level
        const color = getVolatilityColor(vol, colorScheme);
        colors.push(color.r, color.g, color.b);
      }
    }
    
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return geo;
  }, [data, strikes, expirations, colorScheme]);
  
  // Animation
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.1) * 0.05;
    }
  });
  
  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial vertexColors wireframe={false} />
    </mesh>
  );
};

// Grid Lines Component
const GridLines: React.FC<{
  strikes: number[];
  expirations: number[];
}> = ({ strikes, expirations }) => {
  const lines = useMemo(() => {
    const lineGeometry = [];
    
    // Strike lines
    strikes.forEach((strike, i) => {
      const x = (i / (strikes.length - 1)) * 10 - 5;
      lineGeometry.push([
        [x, -5, 0],
        [x, 5, 0]
      ]);
    });
    
    // Expiration lines
    expirations.forEach((exp, i) => {
      const y = (i / (expirations.length - 1)) * 10 - 5;
      lineGeometry.push([
        [-5, y, 0],
        [5, y, 0]
      ]);
    });
    
    return lineGeometry;
  }, [strikes, expirations]);
  
  return (
    <group>
      {lines.map((line, index) => (
        <Line
          key={index}
          points={line}
          color="#666666"
          lineWidth={1}
          opacity={0.3}
        />
      ))}
    </group>
  );
};

// Axis Labels Component
const AxisLabels: React.FC<{
  strikes: number[];
  expirations: number[];
}> = ({ strikes, expirations }) => {
  return (
    <group>
      {/* Strike labels */}
      {strikes.map((strike, i) => (
        <Text
          key={`strike-${i}`}
          position={[(i / (strikes.length - 1)) * 10 - 5, -6, 0]}
          fontSize={0.3}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {strike.toFixed(0)}
        </Text>
      ))}
      
      {/* Expiration labels */}
      {expirations.map((exp, i) => (
        <Text
          key={`exp-${i}`}
          position={[-6, (i / (expirations.length - 1)) * 10 - 5, 0]}
          fontSize={0.3}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {exp.toFixed(1)}
        </Text>
      ))}
      
      {/* Axis titles */}
      <Text
        position={[0, -7, 0]}
        fontSize={0.5}
        color="cyan"
        anchorX="center"
        anchorY="middle"
      >
        Strike Price
      </Text>
      
      <Text
        position={[-7, 0, 0]}
        fontSize={0.5}
        color="cyan"
        anchorX="center"
        anchorY="middle"
        rotation={[0, 0, Math.PI / 2]}
      >
        Time to Expiration
      </Text>
      
      <Text
        position={[0, 0, 8]}
        fontSize={0.5}
        color="cyan"
        anchorX="center"
        anchorY="middle"
      >
        Implied Volatility
      </Text>
    </group>
  );
};

// Color mapping function
function getVolatilityColor(volatility: number, scheme: string): THREE.Color {
  const normalized = Math.min(Math.max(volatility / 100, 0), 1); // Normalize to 0-1
  
  switch (scheme) {
    case 'thermal':
      return new THREE.Color().setHSL(
        (1 - normalized) * 0.7, // Hue from red to blue
        1.0, // Full saturation
        0.5 + normalized * 0.3 // Lightness
      );
    
    case 'rainbow':
      return new THREE.Color().setHSL(
        normalized * 0.8, // Rainbow hue
        1.0,
        0.6
      );
    
    case 'grayscale':
      const gray = normalized * 0.8 + 0.2;
      return new THREE.Color(gray, gray, gray);
    
    default:
      return new THREE.Color(0.5, 0.5, 0.5);
  }
}

export const VolatilitySurface3D: React.FC<VolatilitySurfaceProps> = ({
  volatilityData,
  underlyingPrice,
  riskFreeRate,
  showGrid = true,
  colorScheme = 'thermal',
}) => {
  const [surfaceOpacity, setSurfaceOpacity] = useState(0.8);
  const [showWireframe, setShowWireframe] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  
  // Statistics calculation
  const stats = useMemo(() => {
    const flatVols = volatilityData.impliedVols.flat();
    return {
      min: Math.min(...flatVols),
      max: Math.max(...flatVols),
      avg: flatVols.reduce((a, b) => a + b, 0) / flatVols.length,
      atm: volatilityData.impliedVols[Math.floor(volatilityData.expirations.length / 2)][
        Math.floor(volatilityData.strikes.length / 2)
      ],
    };
  }, [volatilityData]);
  
  return (
    <Paper sx={{ 
      p: 3, 
      background: 'linear-gradient(145deg, #1e1e1e 0%, #2d2d2d 100%)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: 2,
      height: '600px'
    }}>
      <Typography variant="h6" gutterBottom>
        Interactive 3D Volatility Surface
      </Typography>
      
      {/* Controls */}
      <Box sx={{ display: 'flex', gap: 3, mb: 2, alignItems: 'center' }}>
        <Box sx={{ minWidth: 120 }}>
          <Typography gutterBottom>Surface Opacity</Typography>
          <Slider
            value={surfaceOpacity}
            onChange={(_, value) => setSurfaceOpacity(value as number)}
            min={0.1}
            max={1}
            step={0.1}
            size="small"
          />
        </Box>
        
        <FormControlLabel
          control={
            <Switch
              checked={showWireframe}
              onChange={(e) => setShowWireframe(e.target.checked)}
              size="small"
            />
          }
          label="Wireframe"
        />
        
        <FormControlLabel
          control={
            <Switch
              checked={showGrid}
              onChange={() => {}}
              size="small"
            />
          }
          label="Grid"
        />
      </Box>
      
      {/* Statistics */}
      <Box sx={{ display: 'flex', gap: 3, mb: 2, fontSize: '0.875rem' }}>
        <Box>Vol Range: {stats.min.toFixed(1)}% - {stats.max.toFixed(1)}%</Box>
        <Box>Average: {stats.avg.toFixed(1)}%</Box>
        <Box>ATM: {stats.atm.toFixed(1)}%</Box>
        <Box>Underlying: ${underlyingPrice}</Box>
      </Box>
      
      {/* 3D Canvas */}
      <Box sx={{ height: 450, width: '100%' }}>
        <Canvas
          camera={{ position: [15, 15, 15], fov: 60 }}
          style={{ background: 'linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%)' }}
        >
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 10]} intensity={1} />
          <pointLight position={[-10, -10, -10]} intensity={0.5} />
          
          {/* Volatility Surface */}
          <VolatilitySurfaceMesh
            data={volatilityData.impliedVols}
            strikes={volatilityData.strikes}
            expirations={volatilityData.expirations}
            colorScheme={colorScheme}
          />
          
          {/* Grid Lines */}
          {showGrid && (
            <GridLines
              strikes={volatilityData.strikes}
              expirations={volatilityData.expirations}
            />
          )}
          
          {/* Axis Labels */}
          <AxisLabels
            strikes={volatilityData.strikes}
            expirations={volatilityData.expirations}
          />
          
          {/* Controls */}
          <OrbitControls 
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            autoRotate={false}
            autoRotateSpeed={animationSpeed}
          />
        </Canvas>
      </Box>
    </Paper>
  );
};

export default VolatilitySurface3D;
