// Global State management
let activeStillId = null;
let activePage = 'live';
let activePreviewTab = 'blended'; // 'raw', 'mask', or 'blended'

// Cached base64 images returned from analysis
let analysisImages = {
    raw: '',
    mask: '',
    blended: ''
};

// Navigation controller
function navigate(targetId) {
    if (targetId === 'still' && !activeStillId) {
        return; // Disabled until capture happens
    }
    
    const navLive = document.getElementById('nav-live');
    const navStill = document.getElementById('nav-still');
    const pageLive = document.getElementById('page-live-content');
    const pageStill = document.getElementById('page-still-content');
    
    if (targetId === 'live') {
        activePage = 'live';
        navLive.classList.add('active');
        navStill.classList.remove('active');
        pageLive.style.display = 'block';
        pageStill.style.display = 'none';
    } else {
        activePage = 'still';
        navLive.classList.remove('active');
        navStill.classList.add('active');
        pageLive.style.display = 'none';
        pageStill.style.display = 'block';
    }
}

// Toggle stream pause on backend
function toggleStreamPause(isPaused) {
    const payload = new FormData();
    payload.append('paused', isPaused ? 'true' : 'false');
    
    const url = window.DjangoUrls ? window.DjangoUrls.toggleStream : '/toggle_stream/';
    return fetch(url, {
        method: "POST",
        body: payload
    })
    .then(res => res.json())
    .then(data => {
        console.log("[SYSTEM] Stream toggled, is_paused =", data.is_paused);
    })
    .catch(err => console.error("[ERROR] Failed to toggle stream pausing:", err));
}

// Capture dynamic still image frame
function captureAndAnalyzeFrame() {
    const snapBtn = document.querySelector('.btn-capture');
    snapBtn.disabled = true;
    snapBtn.innerHTML = "📸 Snapping Image Matrix...";
    
    const url = window.DjangoUrls ? window.DjangoUrls.captureStill : '/capture/';
    fetch(url, {
        method: "POST"
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            activeStillId = data.still_id;
            
            // Enable Nav Tab
            const navStill = document.getElementById('nav-still');
            navStill.classList.remove('disabled');
            
            // Call pause stream background threads
            toggleStreamPause(true).then(() => {
                // Navigate to Still Studio
                navigate('still');
                // Reset tab to blended
                switchPreviewTab('blended');
                // Trigger Analysis
                runAnalysis();
            });
        } else {
            alert("Failed to capture static stream frame: " + data.message);
        }
    })
    .catch(err => {
        console.error("[ERROR] Failed to capture still:", err);
        alert("Connection error during frame snapping capture.");
    })
    .finally(() => {
        snapBtn.disabled = false;
        snapBtn.innerHTML = "📸 Snap & Analyze Still Frame";
    });
}

// Swaps preview tab
function switchPreviewTab(tabId) {
    activePreviewTab = tabId;
    
    // Toggle tab styles
    document.getElementById('tab-raw').classList.remove('active');
    document.getElementById('tab-mask').classList.remove('active');
    document.getElementById('tab-blended').classList.remove('active');
    
    document.getElementById('tab-' + tabId).classList.add('active');
    
    // Update Viewport Image Source from memory cache
    const imgEl = document.getElementById('stillViewportImg');
    if (tabId === 'raw') {
        imgEl.src = "data:image/jpeg;base64," + analysisImages.raw;
    } else if (tabId === 'mask') {
        imgEl.src = "data:image/jpeg;base64," + analysisImages.mask;
    } else {
        imgEl.src = "data:image/jpeg;base64," + analysisImages.blended;
    }
}

// Update Slider Labels
function updateSliderLabel(paramId, value) {
    document.getElementById('lbl-' + paramId).textContent = value;
}

// Debounce wrapper
let debounceTimer;
function debouncedRunAnalysis() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runAnalysis, 350);
}

// Run parametric analysis on the captured frame
function runAnalysis() {
    if (!activeStillId) return;
    
    const loader = document.getElementById('studio-loader');
    loader.style.display = 'flex';
    
    const payload = new FormData();
    payload.append('still_id', activeStillId);
    payload.append('auto_enhance', document.getElementById('param-auto-enhance').checked ? 'true' : 'false');
    payload.append('contrast', document.getElementById('param-contrast').value);
    payload.append('brightness', document.getElementById('param-brightness').value);
    payload.append('bilateral_d', document.getElementById('param-bilateral_d').value);
    payload.append('bilateral_sigma_color', document.getElementById('param-bilateral_sigma_color').value);
    payload.append('bilateral_sigma_space', document.getElementById('param-bilateral_sigma_color').value);
    payload.append('threshold_block_size', document.getElementById('param-threshold_block_size').value);
    payload.append('threshold_c', document.getElementById('param-threshold_c').value);
    payload.append('min_area', document.getElementById('param-min_area').value);
    payload.append('min_perimeter', document.getElementById('param-min_perimeter').value);
    payload.append('min_aspect_ratio', document.getElementById('param-min_aspect_ratio').value);
    
    const url = window.DjangoUrls ? window.DjangoUrls.analyzeCapturedFrame : '/analyze/';
    fetch(url, {
        method: "POST",
        body: payload
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            // Cache images
            analysisImages.raw = data.raw_image;
            analysisImages.mask = data.mask_image;
            analysisImages.blended = data.blended_image;
            
            // Refresh active viewport
            switchPreviewTab(activePreviewTab);
            
            // Update Metrics Counters
            document.getElementById('metric-total-cracks').textContent = data.total_cracks;
            document.getElementById('metric-critical').textContent = data.critical_count;
            document.getElementById('metric-moderate').textContent = data.moderate_count;
            
            // Update Table
            const tbody = document.getElementById('severityTableBody');
            tbody.innerHTML = '';
            
            if (data.cracks.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="empty-row-msg">No structural defects detected with current parameters. Adjust sliders to fine-tune.</td></tr>';
            } else {
                data.cracks.forEach(c => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>Crack #${c.id}</strong></td>
                        <td><span class="badge badge-${c.severity}">${c.severity}</span></td>
                        <td style="font-family: monospace; font-size: 0.8rem; color: var(--text-muted)">(${c.box.join(', ')})</td>
                        <td>${c.width_px} px</td>
                        <td>${c.height_px} px</td>
                        <td>${c.perimeter_px} px</td>
                        <td>${c.area_px} px²</td>
                        <td>${c.elongation}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } else {
            alert("Inspection analysis failed: " + data.message);
        }
    })
    .catch(err => {
        console.error("[ERROR] Failed to run analysis:", err);
    })
    .finally(() => {
        loader.style.display = 'none';
    });
}

// Resume camera feed on backend
function resumeLiveStream() {
    toggleStreamPause(false).then(() => {
        activeStillId = null;
        // Disable studio tab
        const navStill = document.getElementById('nav-still');
        navStill.classList.add('disabled');
        
        // Navigate back to live
        navigate('live');
    });
}

// Camera Stream source controls
function handleSourceChange() {
    const selector = document.getElementById('cameraSelector');
    const customGroup = document.getElementById('customUrlGroup');
    
    if (selector.value === 'custom') {
        customGroup.style.display = 'flex';
    } else {
        customGroup.style.display = 'none';
        fireSourceUpdate(selector.value);
    }
}

// Apply custom source input
function applyCustomSource() {
    const customUrl = document.getElementById('customUrlInput').value.trim();
    if (customUrl) {
        fireSourceUpdate(customUrl);
    }
}

// Dispatch source change to backend
function fireSourceUpdate(sourceValue) {
    const payload = new FormData();
    payload.append('source', sourceValue);

    const url = window.DjangoUrls ? window.DjangoUrls.changeSource : '/change_source/';
    fetch(url, {
        method: "POST",
        body: payload
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            console.log("[SYSTEM] Source updated successfully:", data.active_source);
        } else {
            alert("Failed to transition stream target.");
        }
    })
    .catch(err => console.error("[ERROR] Failed to forward switch signal:", err));
}

// Switch between the processed live feed stream views
function switchLiveTab(tabId) {
    const tabs = ['blended', 'grayscale', 'denoised', 'thresholded', 'morphological'];
    tabs.forEach(t => {
        const tabEl = document.getElementById('live-tab-' + t);
        if (tabEl) {
            if (t === tabId) {
                tabEl.classList.add('active');
            } else {
                tabEl.classList.remove('active');
            }
        }
    });
    
    const imgEl = document.getElementById('liveFeedImg');
    if (!imgEl || !window.DjangoUrls) return;
    
    if (tabId === 'grayscale') {
        imgEl.src = window.DjangoUrls.feedGrayscale;
    } else if (tabId === 'denoised') {
        imgEl.src = window.DjangoUrls.feedDenoised;
    } else if (tabId === 'thresholded') {
        imgEl.src = window.DjangoUrls.feedThresholded;
    } else if (tabId === 'morphological') {
        imgEl.src = window.DjangoUrls.feedMorphological;
    } else {
        imgEl.src = window.DjangoUrls.feedBlended;
    }
}
