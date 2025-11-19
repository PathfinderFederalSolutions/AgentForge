'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

/**
 * Day-mode 3D background: wireframe terrain + subtle particle drift.
 * Night mode: toned down, red-on-black grid with reduced motion.
 */
export default function TopoStage() {
  const mountRef = useRef<HTMLDivElement>(null);
  const snap = useSnapshot(store);

  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth / mount.clientHeight, 1, 5000);
    camera.position.set(0, 120, 260);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const resize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    // wireframe terrain
    const grid = new THREE.GridHelper(2000, 80, 0x113355, 0x0d1a2b);
    (grid.material as THREE.Material).opacity = snap.theme === 'night' ? 0.15 : 0.25;
    (grid.material as THREE.Material).transparent = true;
    grid.position.y = -40;
    scene.add(grid);

    // topographic sine surface
    const geo = new THREE.PlaneGeometry(2000, 1000, 180, 90);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      pos.setZ(i, 12 * Math.sin(x / 42) + 8 * Math.cos(y / 36));
    }
    geo.computeVertexNormals();
    const mat = new THREE.MeshBasicMaterial({
      color: snap.theme === 'night' ? 0x551111 : 0x0f2438,
      wireframe: true,
      transparent: true,
      opacity: snap.theme === 'night' ? 0.18 : 0.22
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2.2;
    mesh.position.y = -20;
    scene.add(mesh);

    // particles
    const pGeo = new THREE.BufferGeometry();
    const count = 1200;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = (Math.random() - 0.5) * 1400;
      positions[i * 3 + 1] = Math.random() * 240;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 700;
    }
    pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const pMat = new THREE.PointsMaterial({
      size: 1.4,
      color: snap.theme === 'night' ? 0xff2b2b : 0x4fc3f7,
      opacity: snap.theme === 'night' ? 0.35 : 0.5,
      transparent: true
    });
    const points = new THREE.Points(pGeo, pMat);
    scene.add(points);

    // ambient tint
    scene.background = null;

    let raf = 0;
    const clock = new THREE.Clock();

    function animate() {
      const t = clock.getElapsedTime();
      mesh.rotation.z = 0.02 * Math.sin(t * 0.2);
      points.rotation.y = t * 0.02;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      mount.removeChild(renderer.domElement);
      renderer.dispose();
      geo.dispose();
      pGeo.dispose();
    };
  }, [snap.theme]);

  return <div ref={mountRef} className="fixed inset-0 -z-10" />;
}
