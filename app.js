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
    fetch('models/MoritzLaurer_mDeBERTa-v3-base-mnli-xnli/model_summary.json')
      .then(res => res.json())
      .then(data => renderDashboard(data))
      .catch(err => console.log("Auto-load requires local HTTP server: " + err.message));
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

    if (data.executable_graph_svg) {
      viewGraphBtn.style.display = 'block';
      graphContainer.innerHTML = data.executable_graph_svg;
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
        shape: `[${tensors[key].shape?.join(', ') || '?' }]`,
        dtype: tensors[key].dtype || '?'
      }
    });
    
    // Sort array by name
    allTensors.sort((a,b) => a.name.localeCompare(b.name));
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

  // Modal logic
  viewGraphBtn.addEventListener('click', () => {
    graphModal.classList.add('active');
  });
  closeModalBtn.addEventListener('click', () => {
    graphModal.classList.remove('active');
  });
  graphModal.addEventListener('click', (e) => {
    if (e.target === graphModal) {
      graphModal.classList.remove('active');
    }
  });
});
