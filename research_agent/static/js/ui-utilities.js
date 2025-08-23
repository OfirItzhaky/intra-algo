/**
 * UI Utilities Module
 * Common UI helper functions and utilities
 */

// RAG Panel Management
function toggleRagPanel(agent) {
  const panel = document.getElementById(`rag-${agent}-panel`);
  if (panel) {
    panel.style.display =
      panel.style.display === 'none' || panel.style.display === '' ? 'block' : 'none';
  }
}

function renderRagMarkdown(content) {
  if (window.marked) {
    return window.marked.parse(content || '');
  }
  // Fallback: replace newlines with <br>
  return (content || '').replace(/\n/g, '<br>');
}

function flashRagPanel(agent) {
  const panel = document.getElementById(`rag-${agent}-panel`);
  if (panel) {
    panel.classList.add('rag-flash');
    setTimeout(() => panel.classList.remove('rag-flash'), 700);
  }
}

// Loading States
function showLoadingState(agent) {
  const output = document.getElementById(`${agent}_output`);
  output.classList.add('fade-out');
  // Insert spinner/shimmer at top if not present
  let spinnerId = `${agent}-spinner`;
  if (!document.getElementById(spinnerId)) {
    const spinner = document.createElement('div');
    spinner.id = spinnerId;
    spinner.innerHTML = `<div class="d-flex align-items-center mb-2"><div class="spinner-border text-${agent === 'instinct' ? 'info' : 'success'} me-2" role="status" style="width: 1.5rem; height: 1.5rem;"></div><span>Running...</span></div>`;
    output.parentNode.insertBefore(spinner, output);
  }
}

function hideLoadingState(agent) {
  const output = document.getElementById(`${agent}_output`);
  output.classList.remove('fade-out');
  output.classList.add('fade-in');
  setTimeout(() => output.classList.remove('fade-in'), 600);
  let spinner = document.getElementById(`${agent}-spinner`);
  if (spinner) spinner.remove();
}

function setButtonLoading(btn, isLoading) {
  if (isLoading) {
    btn.disabled = true;
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'Running...';
  } else {
    btn.disabled = false;
    if (btn.dataset.originalText) btn.textContent = btn.dataset.originalText;
  }
}

// Enhancement Button Management
function enableEnhanceButton(agent) {
  const btn = document.getElementById(`enhance-${agent}-btn`);
  if (btn) btn.disabled = false;
  const inputs = document.getElementById(`enhance-${agent}-inputs`);
  if (inputs) inputs.style.display = 'block';
}

// Image Preview Management
function addImageToPreview(file, dropzone, previewArea) {
  const reader = new FileReader();
  reader.onload = function (e) {
    const imgWrapper = document.createElement('div');
    imgWrapper.style.cssText = 'position: relative; display: inline-block;';

    const img = document.createElement('img');
    img.src = e.target.result;
    img.style.cssText =
      'width: 80px; height: 80px; object-fit: cover; border-radius: 6px; border: 2px solid #dee2e6;';

    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '×';
    removeBtn.style.cssText =
      'position: absolute; top: -8px; right: -8px; width: 20px; height: 20px; border-radius: 50%; background: #dc3545; color: white; border: none; font-size: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center;';
    removeBtn.onclick = () => imgWrapper.remove();

    imgWrapper.appendChild(img);
    imgWrapper.appendChild(removeBtn);
    previewArea.appendChild(imgWrapper);
  };
  reader.readAsDataURL(file);
}

// Clipboard paste handling
function handlePaste(e, dropzone, previewArea) {
  const items = e.clipboardData?.items;
  if (!items) return;

  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image') !== -1) {
      const file = items[i].getAsFile();
      addImageToPreview(file, dropzone, previewArea);
    }
  }
}

// Form Data Management
function getFormData() {
  const formData = new FormData();
  const maxRiskInput = document.getElementById('max_risk');
  const maxRiskPerTradeInput = document.getElementById('max_risk_per_trade');

  if (maxRiskInput) formData.append('max_risk', maxRiskInput.value);
  if (maxRiskPerTradeInput) formData.append('max_risk_per_trade', maxRiskPerTradeInput.value);

  return formData;
}

// LLM Cost Tracking
let totalTokens = 0;
let totalCost = 0;
const TOKEN_RATE = 0.0015 / 1000; // $0.0015 per 1K tokens

function addTokens(tokens) {
  if (!tokens || isNaN(tokens)) return;
  totalTokens += tokens;
  totalCost = totalTokens * TOKEN_RATE;

  const llmTokensElem = document.getElementById('llm-tokens');
  const llmCostElem = document.getElementById('llm-cost');

  if (llmTokensElem) llmTokensElem.textContent = totalTokens.toLocaleString();
  if (llmCostElem) llmCostElem.textContent = '$' + totalCost.toFixed(4);
}

// Format optimization markdown for display
function formatOptimizationMarkdown(raw) {
  function escapeHtml(s) {
    return (s || '').replace(
      /[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]
    );
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
      .replace(/^###\s*(.*)$/gim, '<h4>$1</h4>')
      .replace(/^##\s*(.*)$/gim, '<h3>$1</h3>')
      .replace(/^#\s*(.*)$/gim, '<h2>$1</h2>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }
  return '<div class="llm-enhanced">' + htmlBody + '</div>';
}

/**
 * Attach correct handlers to a dropzone and file input:
 * - Click: opens file picker
 * - Paste: handles image paste, does NOT open file picker
 * Usage: UIUtilities.setupImageDropzone(dropzoneElem, fileInputElem, previewAreaElem)
 */
function setupImageDropzone(dropzone, fileInput, previewArea) {
  if (!dropzone || !fileInput) return;
  // Click: open file picker
  dropzone.addEventListener('click', function(e) {
    // Only open file picker if not focused by a paste event
    fileInput.click();
  });
  // Paste: handle image paste, do NOT open file picker
  dropzone.addEventListener('paste', function(e) {
    e.preventDefault();
    handlePaste(e, dropzone, previewArea);
    // DO NOT call fileInput.click() here!
  });
}

// Export functions for global use
window.UIUtilities = {
  toggleRagPanel,
  renderRagMarkdown,
  flashRagPanel,
  showLoadingState,
  hideLoadingState,
  setButtonLoading,
  enableEnhanceButton,
  addImageToPreview,
  handlePaste,
  getFormData,
  addTokens,
  formatOptimizationMarkdown,
  setupImageDropzone,
};

// Also expose individual functions globally for onclick handlers
window.toggleRagPanel = toggleRagPanel;
window.formatOptimizationMarkdown = formatOptimizationMarkdown;
