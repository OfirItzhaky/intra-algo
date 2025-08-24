/**
 * Main JavaScript Entry Point
 * Initializes all modules and sets up the scalping agent interface
 */

document.addEventListener('DOMContentLoaded', function () {
  // Initialize all modules
  console.log('Initializing Scalping Agent modules...');

  // Initialize form handlers
  if (window.FormHandlers) {
    window.FormHandlers.initializeFormHandlers();
  }

  // Initialize multi-timeframe agent
  if (window.MultiTimeframeAgent) {
    window.MultiTimeframeAgent.initializeMultiTimeframeAgent();
  }

  // Setup regression strategy defaults hover panel
  setupRegressionDefaults();

  // --- VWAP Dropzone Initialization (exact copy from original working code) ---
  if (document.getElementById('vwap-paste-dropzone')) {
    const vwapDropzone = document.getElementById('vwap-paste-dropzone');
    const vwapPreviewArea = document.getElementById('vwap-multi-image-preview');
    const vwapBtn = document.getElementById('run-vwap-strategy-btn');
    const vwapCsvUploadInput = document.getElementById('vwap-csv-upload');
    let vwapImages = [];

    function updateVwapBtnState() {
      const hasCsv = vwapCsvUploadInput && vwapCsvUploadInput.files && vwapCsvUploadInput.files.length > 0;
      const hasImages = vwapImages.length > 0;
      vwapBtn.disabled = !(hasCsv || hasImages);
    }

    function renderVwapImagePreview() {
      vwapPreviewArea.innerHTML = '';
      vwapImages.forEach((img, idx) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'border rounded d-flex flex-column';
        wrapper.style.cssText = `
          width: 140px; 
          height: 120px; 
          position: relative; 
          overflow: hidden;
          background: #f8f9fa;
          border: 2px solid #dee2e6 !important;
        `;
        
        // Remove button
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.innerHTML = '×';
        removeBtn.style.cssText = `
          position: absolute; 
          top: 4px; 
          right: 4px; 
          width: 24px; 
          height: 24px; 
          border-radius: 50%; 
          background: rgba(220, 53, 69, 0.9); 
          color: white; 
          border: none; 
          font-size: 14px; 
          font-weight: bold;
          cursor: pointer; 
          z-index: 10;
          display: flex; 
          align-items: center; 
          justify-content: center;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        `;
        removeBtn.title = 'Remove image';
        removeBtn.onclick = () => {
          vwapImages.splice(idx, 1);
          renderVwapImagePreview();
          updateVwapBtnState();
        };
        wrapper.appendChild(removeBtn);
        
        // Image container that fills most of the space
        const imgContainer = document.createElement('div');
        imgContainer.style.cssText = `
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 4px;
          overflow: hidden;
        `;
        
        // Image
        const imgElem = document.createElement('img');
        imgElem.src = img.url;
        imgElem.style.cssText = `
          max-width: 100%;
          max-height: 100%;
          width: auto;
          height: auto;
          object-fit: contain;
          border-radius: 4px;
        `;
        imgContainer.appendChild(imgElem);
        wrapper.appendChild(imgContainer);
        
        // Label at bottom
        const label = document.createElement('div');
        label.className = 'text-center px-1';
        label.style.cssText = `
          font-size: 10px;
          color: #6c757d;
          background: rgba(248, 249, 250, 0.9);
          padding: 2px 4px;
          border-top: 1px solid #dee2e6;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        `;
        label.textContent = img.file.name || 'Pasted Image';
        wrapper.appendChild(label);
        
        vwapPreviewArea.appendChild(wrapper);
      });
    }

    // Handle VWAP paste (multiple images) - EXACT COPY FROM ORIGINAL
    vwapDropzone.addEventListener('paste', function(e) {
      console.log('VWAP Paste event triggered');
      let items = (e.clipboardData || window.clipboardData).items;
      for (let i = 0; i < items.length; i++) {
        let item = items[i];
        if (item.type.indexOf('image') !== -1) {
          let blob = item.getAsFile();
          let file = new File([blob], `pasted_vwap_${Date.now()}_${i}.png`, {type: blob.type});
          let reader = new FileReader();
          reader.onload = function(event) {
            vwapImages.push({ file, url: event.target.result });
            renderVwapImagePreview();
            updateVwapBtnState();
            console.log('VWAP Image added, total:', vwapImages.length);
          };
          reader.readAsDataURL(file);
        }
      }
      updateVwapBtnState();
    });

    // Focus dropzone on click for better UX
    vwapDropzone.addEventListener('click', function() {
      vwapDropzone.focus();
      console.log('VWAP Dropzone focused');
    });

    // CSV input change listener
    if (vwapCsvUploadInput) {
      vwapCsvUploadInput.addEventListener('change', function() {
        updateVwapBtnState();
        const vwapCsvSummaryDiv = document.getElementById('vwap-csv-summary');
        if (vwapCsvUploadInput.files.length > 0) {
          const file = vwapCsvUploadInput.files[0];
          const reader = new FileReader();
          reader.onload = function(e) {
            const text = e.target.result;
            const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
            const numRows = Math.max(0, lines.length - 1);
            if (vwapCsvSummaryDiv) {
              vwapCsvSummaryDiv.innerHTML = `📄 CSV Detected: <b>${numRows}</b> bars loaded (${file.name})`;
            }
          };
          reader.readAsText(file);
        } else if (vwapCsvSummaryDiv) {
          vwapCsvSummaryDiv.innerHTML = '';
        }
      });
    }

    // Initial state
    updateVwapBtnState();

    // Store vwapImages globally for VWAP button handler
    window.vwapImages = vwapImages;
  }

  // --- VWAP Renko Agent Button Handler ---
  const vwapRenkoBtn = document.getElementById('run-vwap-renko-agent-btn');
  if (vwapRenkoBtn) {
    console.log('VWAP Renko button found, attaching handler');
    
    // Remove any existing event listeners to prevent conflicts
    vwapRenkoBtn.replaceWith(vwapRenkoBtn.cloneNode(true));
    const newVwapRenkoBtn = document.getElementById('run-vwap-renko-agent-btn');
    
    newVwapRenkoBtn.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      console.log('VWAP Renko button clicked - starting analysis');
      
      // Show loading state
      newVwapRenkoBtn.disabled = true;
      const originalBtnHTML = newVwapRenkoBtn.innerHTML;
      newVwapRenkoBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing...';
      
      // Prepare FormData with same inputs as VWAP agent
      const formData = new FormData();
      
      // Add pasted images from global vwapImages array
      if (window.vwapImages && window.vwapImages.length > 0) {
        console.log(`Adding ${window.vwapImages.length} images to FormData`);
        window.vwapImages.slice(0, 4).forEach(img => formData.append('images', img.file));
      }
      
      // Add selected model (OpenAI vs Gemini)
      const modelRadio = document.querySelector('input[name="llm_model"]:checked');
      if (modelRadio) {
        console.log(`Selected model: ${modelRadio.value}`);
        formData.append('llm_model', modelRadio.value);
      }
      
      // Add optional user notes from session notes field
      const sessionNotes = document.getElementById('session_notes');
      if (sessionNotes && sessionNotes.value.trim()) {
        console.log('Adding user notes to request');
        formData.append('notes', sessionNotes.value.trim());
      }
      
      console.log('Sending request to /run_vwap_renko_agent');
      
      // Send to new endpoint and handle response
      fetch('/run_vwap_renko_agent', { method: 'POST', body: formData })
        .then(response => {
          console.log(`Response status: ${response.status}`);
          
          // Reset button state
          newVwapRenkoBtn.disabled = false;
          newVwapRenkoBtn.innerHTML = originalBtnHTML;
          
          if (response.ok) {
            // Open results in new tab
            const resultWin = window.open('', '_blank');
            if (resultWin) {
              console.log('Opening results in new tab');
              return response.text().then(html => {
                resultWin.document.open();
                resultWin.document.write(html);
                resultWin.document.close();
              });
            } else {
              console.log('Popup blocked, showing inline');
              // Fallback if popup blocked
              return response.text().then(html => {
                const resultDiv = document.createElement('div');
                resultDiv.innerHTML = '<div class="alert alert-warning mt-2">Popup blocked. Results opened inline:</div>' + html;
                document.body.appendChild(resultDiv);
              });
            }
          } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        })
        .catch(err => {
          console.error('VWAP Renko Agent Error:', err);
          // Reset button and show error
          newVwapRenkoBtn.disabled = false;
          newVwapRenkoBtn.innerHTML = originalBtnHTML;
          alert(`VWAP Renko Agent Error: ${err.message}`);
        });
    });
  } else {
    console.warn('VWAP Renko button not found');
  }

  console.log('All modules initialized successfully');
});

// Setup regression defaults hover panel
function setupRegressionDefaults() {
  const regBtn = document.getElementById('run-regression-strategy-btn');
  const hoverPanel = document.getElementById('regression-defaults-hover-panel');
  const contentDiv = document.getElementById('regression-defaults-content');

  if (!regBtn || !hoverPanel || !contentDiv) return;

  let loaded = false;
  let defaultsData = {};

  const keyMap = {
    tick_value: 'Tick Value',
    slippage: 'Slippage',
    min_dist: 'Min Distance',
    max_dist: 'Max Distance',
    min_classifier_signals: 'Min Classifier Signals',
  };

  function renderDefaults(data) {
    let html = '<table class="table table-sm mb-0">';
    for (const [k, v] of Object.entries(data)) {
      const label = keyMap[k] || k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      let value = v;
      if (k === 'tick_value') value = `$${v}`;
      html += `<tr><td style='font-weight:500;'>${label}</td><td style='text-align:right;'>${value}</td></tr>`;
    }
    html += '</table>';
    contentDiv.innerHTML = html;
  }

  regBtn.addEventListener('mouseenter', function () {
    hoverPanel.style.display = 'block';
    if (!loaded) {
      fetch('/regression_strategy_defaults')
        .then(r => r.json())
        .then(data => {
          defaultsData = data;
          renderDefaults(data);
          loaded = true;
        })
        .catch(() => {
          contentDiv.innerHTML = '<span class="text-danger">Failed to load defaults.</span>';
        });
    }
  });

  regBtn.addEventListener('mouseleave', function (e) {
    // Hide only if mouse is not over the panel
    setTimeout(() => {
      if (!hoverPanel.matches(':hover')) {
        hoverPanel.style.display = 'none';
      }
    }, 100);
  });

  hoverPanel.addEventListener('mouseleave', function () {
    hoverPanel.style.display = 'none';
  });

  hoverPanel.addEventListener('mouseenter', function () {
    hoverPanel.style.display = 'block';
  });
}

// Global exports for backwards compatibility
window.ScalpingAgent = {
  setupRegressionDefaults,
};
