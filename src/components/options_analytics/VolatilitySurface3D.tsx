import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';

interface VolatilitySurfaceProps {
  symbol: string;
  data?: VolatilityData[];
  animate?: boolean;
}

interface VolatilityData {
  strike: number;
  expiry: number;
  impliedVol: number;
  moneyness: number;
  timeToExpiry: number;
}

const VolatilitySurface3D: React.FC<VolatilitySurfaceProps> = ({
  symbol,
  data = [],
  animate = true
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera, gl } = useThree();

  // Generate surface geometry from volatility data
  const { geometry, material } = useMemo(() => {
    if (!data.length) {
      return {
        geometry: new THREE.PlaneGeometry(10, 10, 50, 50),
        material: new THREE.MeshPhongMaterial({ 
          color: 0x00ff88, 
          wireframe: true,
          transparent: true,
          opacity: 0.8 
        })
      };
    }

    // Create surface from volatility data
    const width = 50;
    const height = 50;
    const geometry = new THREE.PlaneGeometry(10, 10, width - 1, height - 1);
    
    // Map data to geometry vertices
    const vertices = geometry.attributes.position;
    const colors = new Float32Array(vertices.count * 3);
    
    for (let i = 0; i < vertices.count; i++) {
      const x = vertices.getX(i);
      const y = vertices.getY(i);
      
      // Interpolate volatility value
      const vol = interpolateVolatility(x, y, data);
      vertices.setZ(i, vol * 2); // Scale Z for visibility
      
      // Color based on volatility level
      const color = new THREE.Color();
      color.setHSL(0.7 - vol * 0.7, 1.0, 0.5);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide
    });

    return { geometry, material };
  }, [data]);

  // Animation frame
  useFrame((state) => {
    if (animate && meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  const interpolateVolatility = (x: number, y: number, data: VolatilityData[]) => {
    // Simple bilinear interpolation
    if (!data.length) return 0.2;
    
    // Find closest data points
    const closest = data.reduce((prev, curr) => {
      const prevDist = Math.sqrt(
        Math.pow(prev.moneyness - x, 2) + Math.pow(prev.timeToExpiry - y, 2)
      );
      const currDist = Math.sqrt(
        Math.pow(curr.moneyness - x, 2) + Math.pow(curr.timeToExpiry - y, 2)
      );
      return currDist < prevDist ? curr : prev;
    });

    return closest.impliedVol;
  };

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} material={material} />
      
      {/* Axes */}
      <Line
        points={[[-5, 0, 0], [5, 0, 0]]}
        color="red"
        lineWidth={2}
      />
      <Line
        points={[[0, -5, 0], [0, 5, 0]]}
        color="green"
        lineWidth={2}
      />
      <Line
        points={[[0, 0, 0], [0, 0, 3]]}
        color="blue"
        lineWidth={2}
      />
      
      {/* Labels */}
      <Text
        position={[5.5, 0, 0]}
        color="red"
        fontSize={0.3}
        anchorX="center"
        anchorY="middle"
      >
        Moneyness
      </Text>
      <Text
        position={[0, 5.5, 0]}
        color="green"
        fontSize={0.3}
        anchorX="center"
        anchorY="middle"
      >
        Time to Expiry
      </Text>
      <Text
        position={[0, 0, 3.5]}
        color="blue"
        fontSize={0.3}
        anchorX="center"
        anchorY="middle"
      >
        Implied Vol
      </Text>
    </group>
  );
};

export default VolatilitySurface3D;
