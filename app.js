/*
 * Permission to use, copy, modify, and/or distribute this software for
 * any purpose with or without fee is hereby granted.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
 * FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
 * AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const uploadScreen = document.getElementById('upload-screen');
  const dashboardScreen = document.getElementById('dashboard-screen');

  // Dashboard elements
  const modelTypeBadge = document.getElementById('model-type-badge');
  const summaryStats = document.getElementById('summary-stats');
  const archFlow = document.getElementById('architecture-flow');
  const tensorList = document.getElementById('tensor-list');
  const tensorSearch = document.getElementById('tensor-search');

  // Modal elements
  const viewGraphBtn = document.getElementById('view-graph-btn');
  const graphModal = document.getElementById('graph-modal');
  const closeModalBtn = document.getElementById('close-modal-btn');
  const graphContainer = document.getElementById('graph-container');

  let allTensors = [];

  // File Upload Logic
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');

    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.name.endsWith('.json')) {
      alert("Please upload a valid model_summary.json file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const payload = JSON.parse(e.target.result);
        renderDashboard(payload);
      } catch (err) {
        alert("Error parsing JSON: " + err.message);
      }
    };
    reader.readAsText(file);
  }

  // URL param to auto-load for testing (requires local server)
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.has('test')) {
    const testCandidates = [
      'models/MoritzLaurer_mDeBERTa-v3-base-mnli-xnli/model_summary.json',
      'mock/model_summary.json'
    ];

    const tryFetchTestData = async () => {
      for (const path of testCandidates) {
        try {
          const res = await fetch(path);
          if (!res.ok) continue;
          const data = await res.json();
          renderDashboard(data);
          console.log(`Auto-loaded test data from ${path}`);
          return;
        } catch (_) {
          // Try the next candidate.
        }
      }
      console.log("Auto-load requires local HTTP server and test JSON at models/... or mock/model_summary.json");
    };

    tryFetchTestData();
  }

  // Render Logic
  function renderDashboard(data) {
    // 1. Switch Screen Transition
    uploadScreen.classList.remove('active');
    setTimeout(() => {
      dashboardScreen.classList.add('active');
    }, 300);

    // 2. Summary & Header
    modelTypeBadge.textContent = (data.summary.model_type || "Unknown").toUpperCase();

    summaryStats.innerHTML = ''; // clear
    const statsToRender = [
      { label: 'Vocab Size', value: data.summary.vocab_size?.toLocaleString() || 'N/A' },
      { label: 'Hidden Size', value: data.summary.hidden_size?.toLocaleString() || 'N/A' },
      { label: 'Attn Heads', value: data.summary.num_attention_heads?.toLocaleString() || 'N/A' },
      { label: 'Layers', value: data.summary.num_hidden_layers?.toLocaleString() || 'N/A' },
      { label: 'Max Context', value: data.summary.max_position_embeddings?.toLocaleString() || 'N/A' }
    ];

    statsToRender.forEach((s, idx) => {
      const div = document.createElement('div');
      div.className = `stat-box ${idx === 0 ? 'highlight' : ''}`;
      div.innerHTML = `
        <div class="stat-label">${s.label}</div>
        <div class="stat-value">${s.value}</div>
      `;
      summaryStats.appendChild(div);
    });

    if (data.executable_graph_3d) {
      viewGraphBtn.style.display = 'block';
      init3DGraph(data);
    } else {
      viewGraphBtn.style.display = 'none';
      graphContainer.innerHTML = '';
    }

    // 3. Architecture Flow
    archFlow.innerHTML = '';
    const layers = data.layer_names || [];
    layers.forEach((layer) => {
      const node = document.createElement('div');
      node.className = 'arch-node';
      node.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 0.25rem;">${layer.description || layer.id}</div>
        <div style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">${layer.id}</div>
      `;

      // If Sub-components exist, render them inside the accordion
      if (layer.sub_components && layer.sub_components.length > 0) {
        const subContainer = document.createElement('div');
        subContainer.className = 'sub-components';
        layer.sub_components.forEach(sub => {
          const subEl = document.createElement('div');
          subEl.className = 'sub-item';
          subEl.textContent = sub;
          subContainer.appendChild(subEl);
        });

        node.appendChild(subContainer);
        // Toggle on click
        node.addEventListener('click', () => {
          node.classList.toggle('expanded');
        });
      }
      archFlow.appendChild(node);
    });

    // 4. Tensor Inventory
    const tensors = data.tensor_names || {};
    allTensors = Object.keys(tensors).map(key => {
      return {
        name: key,
        shape: `[${tensors[key].shape?.join(', ') || '?'}]`,
        dtype: tensors[key].dtype || '?'
      }
    });

    // Sort array by name
    allTensors.sort((a, b) => a.name.localeCompare(b.name));
    renderTensors(allTensors);
  }

  // Render Tensor List
  function renderTensors(list) {
    tensorList.innerHTML = '';
    list.forEach(t => {
      const row = document.createElement('div');
      row.className = 'tensor-row';
      row.innerHTML = `
        <div class="tensor-name">${t.name}</div>
        <div class="tensor-meta">
          <span class="tensor-shape">${t.shape}</span>
          <span class="tensor-dtype">${t.dtype}</span>
        </div>
      `;
      tensorList.appendChild(row);
    });
  }

  // Tensor Search/Filter
  tensorSearch.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = allTensors.filter(t => t.name.toLowerCase().includes(term));
    renderTensors(filtered);
  });

  viewGraphBtn.addEventListener('click', () => {
    graphModal.classList.add('active');
    // Force renderer resize and update when modal becomes visible
    if (renderer && camera) {
      setTimeout(() => {
        camera.aspect = graphContainer.clientWidth / graphContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(graphContainer.clientWidth, graphContainer.clientHeight);
      }, 50);
    }
  });
  closeModalBtn.addEventListener('click', () => {
    graphModal.classList.remove('active');
  });
  graphModal.addEventListener('click', (e) => {
    if (e.target === graphModal) {
      graphModal.classList.remove('active');
    }
  });

  // 3D Rendering Logic
  let renderer, scene, camera, controls;
  let layerSprites = [];
  let layerBoxes = [];
  let animationTime = 0;

  function init3DGraph(data) {
    const graphData = data.executable_graph_3d;
    if (!renderer) {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1c23); // Match existing styling lightly

      camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10000);
      camera.position.set(0, 0, 100);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      graphContainer.appendChild(renderer.domElement);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      window.addEventListener('resize', () => {
        if (!graphModal.classList.contains('active')) return;
        camera.aspect = graphContainer.clientWidth / graphContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(graphContainer.clientWidth, graphContainer.clientHeight);
      });

      // render loop
      renderer.setAnimationLoop(() => {
        if (graphModal.classList.contains('active')) {
          controls.update();

          // Apply animations
          animationTime += 0.016;
          layerSprites.forEach(sprite => {
            if (sprite.userData && sprite.userData.baseY !== undefined) {
              sprite.position.y = sprite.userData.baseY + Math.sin(animationTime * 2 + sprite.userData.offset) * 1.5;
            }
          });
          layerBoxes.forEach(box => {
            if (box.material && box.userData && box.userData.baseOpacity !== undefined) {
              box.material.opacity = box.userData.baseOpacity + Math.sin(animationTime + box.userData.offset) * 0.03;
            }
          });

          renderer.render(scene, camera);
        }
      });
    }

    // Clear previous scene
    while (scene.children.length > 0) {
      scene.remove(scene.children[0]);
    }

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(200, 500, 300);
    scene.add(dirLight);

    const { nodes, edges } = graphData;

    // 1. Nodes using InstancedMesh
    const geometry = new THREE.SphereGeometry(0.5, 16, 16);
    const material = new THREE.MeshPhongMaterial({ shininess: 30 });
    const instancedMesh = new THREE.InstancedMesh(geometry, material, nodes.length);

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    let center = new THREE.Vector3(0, 0, 0);

    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      dummy.position.set(n.pos[0], n.pos[1], n.pos[2]);
      center.add(dummy.position);
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);

      color.setRGB(n.color[0], n.color[1], n.color[2]);
      instancedMesh.setColorAt(i, color);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
    scene.add(instancedMesh);

    if (nodes.length > 0) {
      center.divideScalar(nodes.length);
      controls.target.copy(center);
      camera.position.set(center.x, center.y, center.z + 50);
    }

    // 2. Edges using separate LineSegments by length to manage opacity
    const edgePointsShort = [];
    const edgePointsMedium = [];
    const edgePointsLong = [];

    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      if (e.points && e.points.length >= 2) {
        const dy = Math.abs(e.points[1][1] - e.points[0][1]);
        const pts = [
          e.points[0][0], e.points[0][1], e.points[0][2],
          e.points[1][0], e.points[1][1], e.points[1][2]
        ];

        if (dy > 20.0) {
          edgePointsLong.push(...pts);
        } else if (dy > 6.0) {
          edgePointsMedium.push(...pts);
        } else {
          edgePointsShort.push(...pts);
        }
      }
    }

    // Short connections (opaque)
    if (edgePointsShort.length > 0) {
      const edgeGeomS = new THREE.BufferGeometry();
      edgeGeomS.setAttribute('position', new THREE.Float32BufferAttribute(edgePointsShort, 3));
      const edgeMatS = new THREE.LineBasicMaterial({ color: 0x4f5b66, transparent: true, opacity: 0.6 });
      scene.add(new THREE.LineSegments(edgeGeomS, edgeMatS));
    }

    // Medium connections (semi-transparent)
    if (edgePointsMedium.length > 0) {
      const edgeGeomM = new THREE.BufferGeometry();
      edgeGeomM.setAttribute('position', new THREE.Float32BufferAttribute(edgePointsMedium, 3));
      const edgeMatM = new THREE.LineBasicMaterial({ color: 0x4f5b66, transparent: true, opacity: 0.15 });
      scene.add(new THREE.LineSegments(edgeGeomM, edgeMatM));
    }

    // Long connections (barely visible)
    if (edgePointsLong.length > 0) {
      const edgeGeomL = new THREE.BufferGeometry();
      edgeGeomL.setAttribute('position', new THREE.Float32BufferAttribute(edgePointsLong, 3));
      const edgeMatL = new THREE.LineBasicMaterial({ color: 0x4f5b66, transparent: true, opacity: 0.02 });
      scene.add(new THREE.LineSegments(edgeGeomL, edgeMatL));
    }

    // 3. Render Conceptual Layer Bounding Boxes and Labels
    layerSprites.forEach(s => scene.remove(s));
    layerBoxes.forEach(b => scene.remove(b));
    layerSprites = [];
    layerBoxes = [];
    
    if (data.node_assignments && data.layer_names) {
      const layerBounds = {};
      const padding = 2.0;

      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const assignment = data.node_assignments[n.id];
        if (assignment && assignment.layer_id !== "unassigned") {
          const layerId = assignment.layer_id;
          if (!layerBounds[layerId]) {
            layerBounds[layerId] = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
          }
          for (let j = 0; j < 3; j++) {
            if (n.pos[j] < layerBounds[layerId].min[j]) layerBounds[layerId].min[j] = n.pos[j];
            if (n.pos[j] > layerBounds[layerId].max[j]) layerBounds[layerId].max[j] = n.pos[j];
          }
        }
      }

      data.layer_names.forEach((layer, idx) => {
        const bounds = layerBounds[layer.id];
        if (!bounds) return;

        for (let j = 0; j < 3; j++) {
          if (bounds.max[j] - bounds.min[j] < 0.1) {
            bounds.max[j] += 1;
            bounds.min[j] -= 1;
          }
        }

        const width = bounds.max[0] - bounds.min[0] + padding * 2;
        const height = bounds.max[1] - bounds.min[1] + padding * 2;
        const depth = bounds.max[2] - bounds.min[2] + padding * 2;

        const centerX = (bounds.min[0] + bounds.max[0]) / 2;
        const centerY = (bounds.min[1] + bounds.max[1]) / 2;
        const centerZ = (bounds.min[2] + bounds.max[2]) / 2;

        const boxGeom = new THREE.BoxGeometry(width, height, depth);
        const color = new THREE.Color().setHSL((idx * 0.6180339887) % 1, 0.6, 0.4); // Golden ratio for distributed coloring
        
        const boxMat = new THREE.MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: 0.08,
          depthWrite: false,
          side: THREE.BackSide
        });
        const boxMesh = new THREE.Mesh(boxGeom, boxMat);
        boxMesh.position.set(centerX, centerY, centerZ);
        boxMesh.userData = { offset: idx, baseOpacity: 0.08 };
        scene.add(boxMesh);
        layerBoxes.push(boxMesh);

        const edgesGeom = new THREE.EdgesGeometry(boxGeom);
        const edgesMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.3 });
        const lineMesh = new THREE.LineSegments(edgesGeom, edgesMat);
        lineMesh.position.set(centerX, centerY, centerZ);
        scene.add(lineMesh);
        layerBoxes.push(lineMesh);

        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#' + color.getHexString();
        ctx.beginPath();
        ctx.roundRect(0, canvas.height/2 - 30, canvas.width, 60, 15);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 36px "Outfit", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(layer.description || layer.id, canvas.width / 2, canvas.height / 2);

        const tex = new THREE.CanvasTexture(canvas);
        tex.minFilter = THREE.LinearFilter;
        
        const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.renderOrder = 999;
        
        // Dynamic sprite sizing based on actual bounds width
        let spriteScaleX = width > 15 ? width : 15;
        let spriteScaleY = spriteScaleX * (canvas.height/canvas.width);
        
        sprite.scale.set(spriteScaleX, spriteScaleY, 1);
        sprite.position.set(centerX, bounds.max[1] + (height * 0.1) + padding + 1.5, bounds.max[2] + padding);
        sprite.userData = { baseY: sprite.position.y, offset: idx };

        scene.add(sprite);
        layerSprites.push(sprite);
      });
    }
  }
});
