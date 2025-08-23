/**
 * Multi-Timeframe Agent Module
 * Handles image upload, paste functionality, and multi-timeframe analysis
 */

// Multi-image state
let chartImages = []; // { file: File, url: string, tag: string }
let vwapImages = []; // Similar structure for VWAP images

// Initialize multi-timeframe functionality
function initializeMultiTimeframeAgent() {
  const dropzone = document.getElementById('paste-dropzone');
  const multiPreviewArea = document.getElementById('multi-image-preview');
  const vwapDropzone = document.getElementById('vwap-paste-dropzone');
  const vwapPreviewArea = document.getElementById('vwap-multi-image-preview');
  let chartFileInput = document.getElementById('chart_file');
  const vwapImageInput = document.getElementById('vwap_image_file');

  // Create hidden input if not present
  if (!chartFileInput) {
    const setupForm = document.getElementById('setup-form');
    if (setupForm) {
      chartFileInput = document.createElement('input');
      chartFileInput.type = 'file';
      chartFileInput.id = 'chart_file';
      chartFileInput.name = 'chart_file';
      chartFileInput.style.display = 'none';
      setupForm.prepend(chartFileInput);
    }
  }

  // Setup dropzone functionality
  if (dropzone) {
    setupDropzone(dropzone, multiPreviewArea, chartImages, chartFileInput);
  }

  if (vwapDropzone && vwapPreviewArea) {
    setupDropzone(vwapDropzone, vwapPreviewArea, vwapImages, vwapImageInput);
  }

  // Setup multi-timeframe agent button
  setupMultiTimeframeButton();

  // Setup VWAP agent button
  setupVWAPButton();
}

function setupDropzone(dropzone, previewArea, imageArray, fileInput) {
  // Handle file input (multiple)
  if (fileInput) {
    fileInput.addEventListener('change', function (e) {
      const files = Array.from(e.target.files);
      files.forEach(file => {
        if (!file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = function (event) {
          imageArray.push({ file, url: event.target.result, tag: '' });
          renderImagePreview(previewArea, imageArray);
        };
        reader.readAsDataURL(file);
      });
      // Reset input so same file can be re-added if removed
      fileInput.value = '';
    });
  }

  // Handle paste (multiple images)
  dropzone.addEventListener('paste', function (e) {
    let items = (e.clipboardData || window.clipboardData).items;
    for (let i = 0; i < items.length; i++) {
      let item = items[i];
      if (item.type.indexOf('image') !== -1) {
        let blob = item.getAsFile();
        let file = new File([blob], `pasted_chart_${Date.now()}_${i}.png`, { type: blob.type });
        let reader = new FileReader();
        reader.onload = function (event) {
          imageArray.push({ file, url: event.target.result, tag: '' });
          renderImagePreview(previewArea, imageArray);
        };
        reader.readAsDataURL(file);
      }
    }
  });

  // Focus/blur handlers for dropzone
  dropzone.addEventListener('focus', function () {
    dropzone.style.borderColor = '#0d6efd';
    dropzone.style.background = '#e7f1ff';
  });

  dropzone.addEventListener('blur', function () {
    dropzone.style.borderColor = '';
    dropzone.style.background = '';
  });

  dropzone.addEventListener('click', function () {
    if (fileInput) fileInput.click();
  });
}

function renderImagePreview(previewArea, imageArray) {
  previewArea.innerHTML = '';
  imageArray.forEach((img, idx) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'border rounded p-2 d-flex flex-column align-items-center';
    wrapper.style.width = '120px';
    wrapper.style.position = 'relative';

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn-close position-absolute top-0 end-0 m-1';
    removeBtn.style.zIndex = 2;
    removeBtn.title = 'Remove image';
    removeBtn.onclick = () => {
      imageArray.splice(idx, 1);
      renderImagePreview(previewArea, imageArray);
    };
    wrapper.appendChild(removeBtn);

    // Image
    const imgElem = document.createElement('img');
    imgElem.src = img.url;
    imgElem.style.maxWidth = '90px';
    imgElem.style.maxHeight = '70px';
    imgElem.className = 'mb-1';
    wrapper.appendChild(imgElem);

    // Timeframe tag dropdown
    const tfSelect = document.createElement('select');
    tfSelect.className = 'form-select form-select-sm mt-1';
    tfSelect.style.width = '90px';
    const timeframeOptions = window.AgentConfig?.TIMEFRAME_OPTIONS || [
      '15m',
      '30m',
      '60m',
      '4h',
      'Daily',
      'Weekly',
    ];
    timeframeOptions.forEach(opt => {
      const option = document.createElement('option');
      option.value = opt;
      option.textContent = opt;
      if (img.tag === opt) option.selected = true;
      tfSelect.appendChild(option);
    });
    tfSelect.value = img.tag || '';
    tfSelect.onchange = e => {
      img.tag = e.target.value;
    };
    wrapper.appendChild(tfSelect);

    // Label
    const label = document.createElement('div');
    label.className = 'text-muted small mt-1';
    label.textContent = img.file.name || 'Pasted Image';
    wrapper.appendChild(label);

    previewArea.appendChild(wrapper);
  });
}

function setupMultiTimeframeButton() {
  const runMultiBtn = document.getElementById('run-multitimeframe-agent-btn');
  const multiResultDiv = document.getElementById('multitimeframe-agent-result');

  if (!runMultiBtn) return;

  runMultiBtn.addEventListener('click', function () {
    const { getFormData, addTokens } = window.UIUtilities;

    if (chartImages.length === 0) {
      if (multiResultDiv) {
        multiResultDiv.innerHTML =
          '<div class="alert alert-warning">Please upload or paste at least one chart image.</div>';
      }
      return;
    }

    // Disable button and show loading
    runMultiBtn.disabled = true;
    const multiBtnOriginal = runMultiBtn.innerHTML;
    runMultiBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing...';

    if (multiResultDiv) multiResultDiv.innerHTML = '';

    // Prepare form data
    const formData = getFormData();
    chartImages.forEach((img, index) => {
      formData.append('chart_images', img.file);
      formData.append(`chart_tag_${index}`, img.tag || '');
    });

    fetch('/run_multitimeframe_agent', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        runMultiBtn.disabled = false;
        runMultiBtn.innerHTML = multiBtnOriginal;

        if (multiResultDiv) {
          let html = '';
          if (data.error) {
            html = `<div class="alert alert-danger">Error: ${data.error}</div>`;
          } else {
            html +=
              '<div class="card mt-3"><div class="card-header bg-success text-white"><b>📊 Multi-Timeframe Analysis Result</b></div><div class="card-body">';

            if (data.summary) {
              html += `<div class="mb-3"><strong>Summary:</strong><br>${data.summary}</div>`;
            }

            if (data.bias_analysis && Array.isArray(data.bias_analysis)) {
              html += '<div class="mb-3"><strong>Bias Analysis:</strong><ul>';
              data.bias_analysis.forEach(bias => {
                html += `<li><strong>${bias.timeframe}:</strong> ${bias.direction} (Confidence: ${bias.confidence}%)</li>`;
              });
              html += '</ul></div>';
            }

            if (data.recommendations) {
              html += `<div class="mb-3"><strong>Recommendations:</strong><br>${data.recommendations}</div>`;
            }

            html += '</div></div>';
          }

          if (data.raw && typeof data.raw === 'string') {
            html += `<div class="alert alert-secondary mt-2"><strong>LLM Output:</strong><pre>${data.raw}</pre></div>`;
          }

          multiResultDiv.innerHTML = html;
        }

        // Add token count if available
        if (data.tokens_used) {
          addTokens(data.tokens_used);
        }
      })
      .catch(err => {
        runMultiBtn.disabled = false;
        runMultiBtn.innerHTML = multiBtnOriginal;
        if (multiResultDiv) {
          multiResultDiv.innerHTML = `<div class='alert alert-danger'>Error: ${err}</div>`;
        }
      });
  });
}

function setupVWAPButton() {
  const vwapBtn = document.getElementById('run-vwap-strategy-btn');
  const vwapResultDiv = document.getElementById('vwap-agent-result');
  const vwapImageInput = document.getElementById('vwap_image_file');

  if (!vwapBtn) return;

  // Add result div if not present
  if (!vwapResultDiv) {
    const newResultDiv = document.createElement('div');
    newResultDiv.id = 'vwap-agent-result';
    vwapBtn.parentNode.appendChild(newResultDiv);
  }

  // Enable VWAP button when images are available
  function updateVWAPButtonState() {
    const hasImages = vwapImages.length > 0 || (vwapImageInput && vwapImageInput.files.length > 0);
    const csvInput = document.getElementById('vwap-csv-upload');
    const hasCSV = csvInput && csvInput.files.length > 0;

    vwapBtn.disabled = !(hasImages || hasCSV);
  }

  // Monitor changes
  if (vwapImageInput) {
    vwapImageInput.addEventListener('change', updateVWAPButtonState);
  }

  const csvInput = document.getElementById('vwap-csv-upload');
  if (csvInput) {
    csvInput.addEventListener('change', updateVWAPButtonState);
  }

  // Initial state check
  updateVWAPButtonState();

  vwapBtn.addEventListener('click', function () {
    if (vwapBtn.disabled) return;

    // Show spinner and running text inside the button
    vwapBtn.disabled = true;
    const originalBtnHTML = vwapBtn.innerHTML;
    vwapBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Running...';
    const resultDiv = document.getElementById('vwap-agent-result');
    if (resultDiv) resultDiv.innerHTML = '<div class="text-muted">Opened results in a new tab…</div>';

    // Open a new tab immediately with a placeholder
    const resultWin = window.open('', '_blank');
    if (!resultWin) {
      // If popup blocked, fallback to inline
      if (resultDiv) resultDiv.innerHTML = '<div class="alert alert-warning">Please allow pop-ups for a better experience. Showing results inline.</div>';
    }

    // Create the skeleton HTML for the popup
    const skeleton = `<!doctype html>
<html><head><meta charset="utf-8"><title>VWAP Agent</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body{font-family:Arial,sans-serif;margin:24px}
  .nav-tabs .nav-link.active{font-weight:600}
  .tab-pane{display:none}
  .tab-pane.active{display:block}
  pre{white-space:pre-wrap}
  .small-muted{color:#6c757d;font-size:0.9rem}
  .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace}
  .llm-enhanced h2,.llm-enhanced h3,.llm-enhanced h4{margin-top:0.8rem}
  .llm-enhanced ul{padding-left:1.2rem}
  .callout{border:1px solid #e6e6ff;border-left:4px solid #7c3aed;background:#f9f7ff;border-radius:8px;padding:12px;margin-bottom:10px}
  .callout.logic{border-left-color:#0d6efd;background:#f5faff}
  .callout.params{border-left-color:#198754;background:#f6fff8}
  .callout-title{font-weight:600;margin-bottom:6px}
</style>
</head><body>
<div class="container">
  <h3 class="mb-3">VWAP Strategy Agent</h3>
  <ul class="nav nav-tabs" id="vwapTabs">
    <li class="nav-item"><button class="nav-link active" id="tab-btn-llm" type="button">LLM Results</button></li>
    <li class="nav-item"><button class="nav-link" id="tab-btn-opt" type="button">Optimization Upload</button></li>
  </ul>
  <div class="tab-content border-start border-end border-bottom p-3">
    <div class="tab-pane active" id="tab-llm">
      <div id="status" class="d-flex align-items-center gap-2 text-secondary">
        <div class="spinner-border text-primary me-2" role="status" aria-hidden="true"></div>
        <div>Thinking…</div>
      </div>
      <div id="content" class="mt-3"></div>
    </div>
    <div class="tab-pane" id="tab-opt">
      <div id="opt-fragment-placeholder" class="text-secondary">Loading optimization UI…</div>
    </div>
  </div>
</div>
</body></html>`;

    if (resultWin) {
      resultWin.document.open();
      resultWin.document.write(skeleton);
      resultWin.document.close();

      // Setup tabs in the popup
      try {
        const d = resultWin.document;
        const btnLlm = d.getElementById('tab-btn-llm');
        const btnOpt = d.getElementById('tab-btn-opt');
        const tabLlm = d.getElementById('tab-llm');
        const tabOpt = d.getElementById('tab-opt');

        const activate = (tab) => {
          if (tab === 'llm') {
            btnLlm.classList.add('active'); btnOpt.classList.remove('active');
            tabLlm.classList.add('active'); tabOpt.classList.remove('active');
          } else {
            btnOpt.classList.add('active'); btnLlm.classList.remove('active');
            tabOpt.classList.add('active'); tabLlm.classList.remove('active');
          }
        };
        
        btnLlm.addEventListener('click', () => activate('llm'));
        btnOpt.addEventListener('click', () => activate('opt'));

        // Load optimization HTML fragment from server and inject into the tab
        fetch('/vwap_optimization_html', { method: 'GET' })
          .then(r => r.text())
          .then(html => {
            const optTab = d.getElementById('tab-opt');
            if (optTab) optTab.innerHTML = html;
            attachOptHandler(d);
          })
          .catch(() => {
            const optTab = d.getElementById('tab-opt');
            if (optTab) optTab.innerHTML = '<div class="text-danger">Failed to load optimization UI.</div>';
          });
      } catch (e) { /* no-op */ }
    }

    // Prepare FormData with up to 4 images
    const formData = new FormData();
    let imagesToSend = vwapImages.length > 0 ? vwapImages.map(img => img.file) : (vwapImageInput && vwapImageInput.files ? Array.from(vwapImageInput.files) : []);
    imagesToSend.slice(0, 4).forEach(file => formData.append('images', file));
    
    const csvInput = document.getElementById('vwap-csv-upload');
    if (csvInput && csvInput.files.length > 0) {
      formData.append('csv_file', csvInput.files[0]);
    }
    
    // Add selected model to FormData
    const modelRadio = document.querySelector('input[name="llm_model"]:checked');
    if (modelRadio) {
      formData.append('llm_model', modelRadio.value);
    }

    fetch('/run_vwap_agent', { method: 'POST', body: formData })
      .then(resp => resp.json())
      .then(data => {
        vwapBtn.disabled = false;
        vwapBtn.innerHTML = originalBtnHTML;

        // Helper to format LLM response for display
        const renderHtml = (payload) => {
          function escapeHtml(s) {
            return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
          }
          
          function formatLlmResponse(raw) {
            let text = (raw || '').trim();
            
            // Extract Strategy Name
            let strategyName = null;
            const nameMatch = text.match(/\*\*\s*Strategy Name:\s*\*\*\s*(.+)/i);
            if (nameMatch) {
              strategyName = nameMatch[1].trim();
              text = text.replace(nameMatch[0], '').trim();
            }
            
            // Extract Execution Logic block
            let executionLogic = null;
            const execMatch = text.match(/\*\*\s*Execution Logic:\s*\*\*\s*([\s\S]+?)(?=\n\s*\n|$)/i);
            if (execMatch) {
              executionLogic = execMatch[1].trim();
              text = text.replace(execMatch[0], '').trim();
            }
            
            // Emoji-enhanced section titles
            text = text.replace(/^###\s*(.*risk.*|.*drawdown.*)$/gim, '### ⚠️ $1');
            text = text.replace(/^##\s*(.*risk.*|.*drawdown.*)$/gim, '## ⚠️ $1');
            text = text.replace(/^###\s*(.*profit.*|.*pnl.*)$/gim, '### 📈 $1');
            text = text.replace(/^##\s*(.*profit.*|.*pnl.*)$/gim, '## 📈 $1');
            text = text.replace(/^###\s*(.*reason.*|.*llm.*)$/gim, '### 🧠 $1');
            text = text.replace(/^##\s*(.*reason.*|.*llm.*)$/gim, '## 🧠 $1');
            
            // Optional: capture Top parameter sets section
            let paramsBlock = null;
            const paramsMatch = text.match(/##\s*Top\s*parameter\s*sets[:\s]*([\s\S]*?)(?=\n\s*##|$)/i);
            if (paramsMatch) {
              paramsBlock = paramsMatch[1].trim();
            }
            
            // Markdown → HTML
            let htmlBody = '';
            if (window.marked && window.marked.parse) {
              htmlBody = window.marked.parse(text);
            } else {
              htmlBody = text
                .replace(/^###\s*(.*)$/gim, '<h4>$1<\/h4>')
                .replace(/^##\s*(.*)$/gim, '<h3>$1<\/h3>')
                .replace(/^#\s*(.*)$/gim, '<h2>$1<\/h2>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1<\/strong>')
                .replace(/\n/g, '<br>');
            }
            
            const style = `
              <style>
                .llm-enhanced h2,.llm-enhanced h3,.llm-enhanced h4{margin-top:0.8rem}
                .llm-enhanced ul{padding-left:1.2rem}
                .callout{border:1px solid #e6e6ff;border-left:4px solid #7c3aed;background:#f9f7ff;border-radius:8px;padding:12px;margin-bottom:10px}
                .callout.logic{border-left-color:#0d6efd;background:#f5faff}
                .callout.params{border-left-color:#198754;background:#f6fff8}
                .callout-title{font-weight:600;margin-bottom:6px}
                .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; white-space: pre-wrap}
                .small-muted{color:#6c757d;font-size:0.9rem}
              </style>`;
              
            let callouts = '';
            if (strategyName) {
              callouts += `<div class="callout"><div class="callout-title">Strategy Name</div><div class="callout-body">${escapeHtml(strategyName)}</div></div>`;
            }
            if (executionLogic) {
              callouts += `<div class="callout logic"><div class="callout-title">Execution Logic</div><div class="callout-body mono">${escapeHtml(executionLogic)}</div></div>`;
            }
            if (paramsBlock) {
              callouts += `<div class="callout params"><div class="callout-title">Top Parameter Sets</div><pre class="mono">${escapeHtml(paramsBlock)}</pre></div>`;
            }
            
            return `${style}<div class='llm-enhanced'>${callouts}<div class='llm-md'>${htmlBody}</div></div>`;
          }
          
          return `
          <div class='card border-primary mt-2'>
            <div class='card-header bg-primary text-white'><b>VWAP Agent LLM Output</b></div>
            <div class='card-body'>
              <div class='mb-2'><b>Model:</b> ${payload.model_name || '-'} | <b>Provider:</b> ${payload.provider || '-'} | <b>Cost:</b> $${payload.llm_cost_usd !== undefined && payload.llm_cost_usd !== null ? Number(payload.llm_cost_usd).toFixed(4) : '-'} | <b>Tokens:</b> ${payload.llm_token_usage || '-'} | <b>Prompt:</b> ${payload.prompt_type === 'single_image' ? 'Single Image' : (payload.prompt_type === 'multi_image' ? '4-Image' : '-') } | <b>Images:</b> ${payload.num_images || 0}</div>
              ${formatLlmResponse(payload.llm_raw_response || '')}
            </div>
          </div>`;
        };

        if (resultWin) {
          const d = resultWin.document;
          const statusDiv = d.getElementById('status');
          const contentDiv = d.getElementById('content');
          if (data.error) {
            statusDiv.innerHTML = `<div class='alert alert-danger mb-0'>${data.error}</div>`;
          } else {
            statusDiv.innerHTML = '';
            contentDiv.innerHTML = renderHtml(data);
          }
        } else {
          // Fallback inline in the same page
          const resultDiv = document.getElementById('vwap-agent-result');
          if (resultDiv) {
            if (data.error) {
              resultDiv.innerHTML = `<div class='alert alert-danger'>${data.error}</div>`;
            } else {
              resultDiv.innerHTML = renderHtml(data);
            }
          }
        }
      })
      .catch(err => {
        vwapBtn.disabled = false;
        vwapBtn.innerHTML = originalBtnHTML;
        if (resultWin) {
          const d = resultWin.document;
          const statusDiv = d.getElementById('status');
          statusDiv.innerHTML = `<div class='alert alert-danger mb-0'>Error: ${err}</div>`;
        } else {
          const resultDiv = document.getElementById('vwap-agent-result');
          if (resultDiv) resultDiv.innerHTML = `<div class='alert alert-danger'>Error: ${err}</div>`;
        }
      });
  });

  // Helper function to attach optimization form handler in popup
  function attachOptHandler(doc) {
    const form = doc.getElementById('opt-form');
    if (!form) { setTimeout(() => attachOptHandler(doc), 200); return; }
    if (form._handlerAttached) return;
    form._handlerAttached = true;
    
    form.addEventListener('submit', function(e){
      e.preventDefault();
      const filesInput = doc.getElementById('opt-files');
      const optStatus = doc.getElementById('opt-status');
      const optOutput = doc.getElementById('opt-output');
      optOutput.innerHTML = '';
      optStatus.innerHTML = '<div class="d-flex align-items-center text-secondary"><div class="spinner-border spinner-border-sm me-2" role="status"></div>Analyzing optimizations…</div>';
      const files = Array.from(filesInput.files || []);
      if (files.length === 0) { optStatus.innerHTML = '<div class="text-danger">Please select at least one .txt file.</div>'; return; }
      if (files.length > 10) { optStatus.innerHTML = '<div class="text-danger">Maximum 10 files allowed.</div>'; return; }
      const fd = new FormData();
      files.forEach(f => fd.append('optimization_files', f));
      fetch('/run_vwap_agent', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(j => {
          if (j.error) { optStatus.innerHTML = `<div class='alert alert-danger'>${j.error}</div>`; return; }
          optStatus.innerHTML = '';
          const pick = j.final_strategy || (j.llm_structured && (j.llm_structured.top_pick || j.llm_structured.final_strategy || (Array.isArray(j.llm_structured.strategy_recommendations) ? j.llm_structured.strategy_recommendations[0] : null)));
          const rationale = j.rationale || (j.llm_structured && (j.llm_structured.rationale || j.llm_structured.reason)) || j.llm_raw_response;
          let out = '';
          if (pick) {
            const name = pick.name || pick.strategy_name || 'Top Strategy';
            const params = Object.entries(pick).filter(([k]) => k !== 'name' && k !== 'strategy_name').map(([k,v]) => `<code class='mono'>${k}</code>: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join('<br>');
            out += `<div class='card border-success mb-3'><div class='card-header bg-success text-white'><b>✅ Selected Strategy</b></div><div class='card-body'><div><b>${name}</b></div><div class='mt-2 small'>${params || 'No parameters available'}</div></div></div>`;
          }
          if (rationale) {
            out += `<div class='card border-secondary'><div class='card-header bg-light'><b>💬 LLM Rationale</b></div><div class='card-body'>${formatOptimizationMarkdown(rationale)}</div></div>`;
          }
          if (j.optimization_cost_metadata) {
            const m = j.optimization_cost_metadata;
            const cost = (m.llm_cost_usd !== undefined && m.llm_cost_usd !== null) ? Number(m.llm_cost_usd).toFixed(4) : '-';
            const tokens = m.llm_token_usage !== undefined && m.llm_token_usage !== null ? m.llm_token_usage : '-';
            const model = m.model_name || '-';
            const prov = m.provider || '-';
            out += `<div class='small text-muted mt-2'>Optimization LLM → Model: ${model} | Provider: ${prov} | Tokens: ${tokens} | Cost: $${cost}</div>`;
          }
          if (!out) {
            out = `<div class='alert alert-info'>No structured optimization result returned. Raw output:</div>${formatOptimizationMarkdown(j.llm_raw_response || '')}`;
          }
          optOutput.innerHTML = out;
        })
        .catch(err => { optStatus.innerHTML = `<div class='alert alert-danger'>Error: ${err}</div>`; });
    });
  }

  // Helper function to format markdown (copied from original)
  function formatOptimizationMarkdown(raw) {
    function escapeHtml(s) {
      return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
    }
    let text = (raw || '').trim();
    // Emoji-enhanced section titles
    text = text.replace(/^###\s*(.*risk.*|.*drawdown.*)$/gim, '### ⚠️ $1');
    text = text.replace(/^##\s*(.*risk.*|.*drawdown.*)$/gim, '## ⚠️ $1');
    text = text.replace(/^###\s*(.*profit.*|.*pnl.*)$/gim, '### 📈 $1');
    text = text.replace(/^##\s*(.*profit.*|.*pnl.*)$/gim, '## 📈 $1');
    text = text.replace(/^###\s*(.*reason.*|.*llm.*)$/gim, '### 🧠 $1');
    text = text.replace(/^##\s*(.*reason.*|.*llm.*)$/gim, '## 🧠 $1');
    let htmlBody = '';
    if (window.marked && window.marked.parse) {
      htmlBody = window.marked.parse(text);
    } else {
      htmlBody = text
        .replace(/^###\s*(.*)$/gim, '<h4>$1<\/h4>')
        .replace(/^##\s*(.*)$/gim, '<h3>$1<\/h3>')
        .replace(/^#\s*(.*)$/gim, '<h2>$1<\/h2>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1<\/strong>')
        .replace(/\n/g, '<br>');
    }
    const style = `
      <style>
        .llm-enhanced h2,.llm-enhanced h3,.llm-enhanced h4{margin-top:0.8rem}
        .llm-enhanced ul{padding-left:1.2rem}
      </style>`;
    return `${style}<div class='llm-enhanced'>${htmlBody}</div>`;
  }
}

// Export functions
window.MultiTimeframeAgent = {
  initializeMultiTimeframeAgent,
  renderImagePreview,
  setupDropzone,
  setupMultiTimeframeButton,
  setupVWAPButton,
  chartImages,
  vwapImages,
};
