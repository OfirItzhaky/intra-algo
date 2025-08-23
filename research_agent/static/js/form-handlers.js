/**
 * Form Handlers Module
 * Event listeners and form-related functionality
 */

// Global context storage
let lastInstinctContext = null;
let lastPlaybookContext = null;

// Initialize form handlers
function initializeFormHandlers() {
  const runInstinctBtn = document.getElementById('run-instinct-btn');
  const runPlaybookBtn = document.getElementById('run-playbook-btn');
  const instinctOutput = document.getElementById('instinct_output');
  const playbookOutput = document.getElementById('playbook_output');
  const csvFileInput = document.getElementById('csv_file');

  // Import utilities from other modules
  const {
    showLoadingState,
    hideLoadingState,
    setButtonLoading,
    enableEnhanceButton,
    flashRagPanel,
    renderRagMarkdown,
    getFormData,
    addTokens,
  } = window.UIUtilities;
  const { renderInstinctAgentResult, renderPlaybookAgentResult, extractTokenCount } =
    window.AgentHandlers;

  // Instinct Agent click handler
  if (runInstinctBtn) {
    runInstinctBtn.addEventListener('click', function () {
      showLoadingState('instinct');
      setButtonLoading(runInstinctBtn, true);
      instinctOutput.innerHTML = '';
      fetch('/start_instinct', {
        method: 'POST',
        body: getFormData(),
      })
        .then(response => response.json())
        .then(data => {
          hideLoadingState('instinct');
          setButtonLoading(runInstinctBtn, false);
          instinctOutput.innerHTML = renderInstinctAgentResult(data);
          addTokens(extractTokenCount(data));
          if (data && data.rag_insights) {
            const ragContent = document.getElementById('rag-instinct-content');
            const ragPanel = document.getElementById('rag-instinct-panel');
            if (ragContent) ragContent.innerHTML = renderRagMarkdown(data.rag_insights);
            if (ragPanel) ragPanel.style.display = 'block';
            flashRagPanel('instinct');
          }
          lastInstinctContext = {
            ...getFormData(),
            result: data,
          };
          enableEnhanceButton('instinct');
        })
        .catch(err => {
          hideLoadingState('instinct');
          setButtonLoading(runInstinctBtn, false);
          instinctOutput.innerHTML = '<div class="alert alert-danger">Error: ' + err + '</div>';
        });
    });
  }

  // Playbook Agent click handler
  if (runPlaybookBtn) {
    runPlaybookBtn.addEventListener('click', function () {
      showLoadingState('playbook');
      setButtonLoading(runPlaybookBtn, true);
      playbookOutput.innerHTML = '';

      // Clear response container before new run
      const responseContainer = document.getElementById('playbook-response-container');
      if (responseContainer) responseContainer.innerHTML = '';

      // Clear bias summary before new run
      const biasSummaryDiv = document.getElementById('bias-summary-block');
      if (biasSummaryDiv) biasSummaryDiv.innerHTML = '';

      fetch('/start_playbook', {
        method: 'POST',
        body: getFormData(),
      })
        .then(response => response.json())
        .then(data => {
          hideLoadingState('playbook');
          setButtonLoading(runPlaybookBtn, false);

          // Show feedback in response container if present
          if (responseContainer) {
            if (data.feedback) {
              responseContainer.innerHTML = `<div class="alert alert-warning"><strong>⚠️ ${data.feedback}</strong></div>`;
            } else {
              responseContainer.innerHTML = '';
            }
          }

          // Render Bias Summary block below response container
          if (biasSummaryDiv) {
            if (Array.isArray(data.bias_summary) && data.bias_summary.length > 0) {
              let html =
                '<h4 class="mt-2 mb-2">🧠 Multi-Timeframe Bias Summary</h4><ul class="mb-2">';
              data.bias_summary.forEach(bias => {
                const conf =
                  bias.confidence !== undefined ? ` (${(bias.confidence * 100).toFixed(0)}%)` : '';
                html += `<li><b>${bias.interval || ''}</b> → ${bias.bias_direction}${conf} – ${bias.reasoning || ''}</li>`;
              });
              html += '</ul>';
              html += `<div id="regression-csv-info" class="mb-2 text-end"></div>`;
              html += `<div id="regression-strategy-result"></div>`;
              biasSummaryDiv.innerHTML = html;
            } else {
              let html =
                '<div class="alert alert-info">No bias summary returned. Showing raw Gemini insights instead:</div>';
              if (Array.isArray(data.raw_bias_data) && data.raw_bias_data.length > 0) {
                data.raw_bias_data.forEach(item => {
                  html += `<pre>${item.reasoning_summary || ''}</pre>`;
                });
              }
              biasSummaryDiv.innerHTML = html;
            }
          }

          playbookOutput.innerHTML = renderPlaybookAgentResult(data);
          addTokens(extractTokenCount(data));

          if (data && data.rag_insights) {
            const ragContent = document.getElementById('rag-playbook-content');
            const ragPanel = document.getElementById('rag-playbook-panel');
            if (ragContent) ragContent.innerHTML = renderRagMarkdown(data.rag_insights);
            if (ragPanel) ragPanel.style.display = 'block';
            flashRagPanel('playbook');
          }

          lastPlaybookContext = {
            ...getFormData(),
            result: data,
          };
          enableEnhanceButton('playbook');

          // Setup regression predictor button logic
          setTimeout(setupRegressionPredictorHandler, 100);
        })
        .catch(err => {
          hideLoadingState('playbook');
          setButtonLoading(runPlaybookBtn, false);
          playbookOutput.innerHTML = '<div class="alert alert-danger">Error: ' + err + '</div>';
        });
    });
  }

  // Chat handlers
  setupChatHandlers();

  // Enhancement handlers
  setupEnhancementHandlers();
}

// Chat functionality
function setupChatHandlers() {
  const instinctQueryBtn = document.getElementById('send_instinct_query');
  const playbookQueryBtn = document.getElementById('send_playbook_query');

  if (instinctQueryBtn) {
    instinctQueryBtn.addEventListener('click', function () {
      const input = document.getElementById('instinct_query_input');
      const question = input.value.trim();
      if (!question) return;

      fetch('/query_instinct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
        .then(response => response.json())
        .then(data => {
          const output = document.getElementById('instinct_output');
          const chat = document.createElement('div');
          chat.className = 'mt-2';
          chat.innerHTML =
            '<div class="alert alert-secondary"><strong>You:</strong> ' +
            question +
            '</div>' +
            '<div class="alert alert-info"><strong>Instinct Agent:</strong> <pre>' +
            JSON.stringify(data, null, 2) +
            '</pre></div>';
          output.appendChild(chat);
          input.value = '';
        });
    });
  }

  if (playbookQueryBtn) {
    playbookQueryBtn.addEventListener('click', function () {
      const input = document.getElementById('playbook_query_input');
      const question = input.value.trim();
      if (!question) return;

      fetch('/query_playbook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
        .then(response => response.json())
        .then(data => {
          const output = document.getElementById('playbook_output');
          const chat = document.createElement('div');
          chat.className = 'mt-2';
          chat.innerHTML =
            '<div class="alert alert-secondary"><strong>You:</strong> ' +
            question +
            '</div>' +
            '<div class="alert alert-success"><strong>Playbook Agent:</strong> <pre>' +
            JSON.stringify(data, null, 2) +
            '</pre></div>';
          output.appendChild(chat);
          input.value = '';
        });
    });
  }
}

// Enhancement functionality
function setupEnhancementHandlers() {
  const enhanceInstinctBtn = document.getElementById('enhance-instinct-btn');
  const enhancePlaybookBtn = document.getElementById('enhance-playbook-btn');

  if (enhanceInstinctBtn) {
    enhanceInstinctBtn.addEventListener('click', function () {
      const { showLoadingState, setButtonLoading } = window.UIUtilities;
      showLoadingState('instinct');
      setButtonLoading(enhanceInstinctBtn, true);
      fetchEnhance('instinct');
    });
  }

  if (enhancePlaybookBtn) {
    enhancePlaybookBtn.addEventListener('click', function () {
      const { showLoadingState, setButtonLoading } = window.UIUtilities;
      showLoadingState('playbook');
      setButtonLoading(enhancePlaybookBtn, true);
      fetchEnhance('playbook');
    });
  }
}

function fetchEnhance(agent) {
  const { hideLoadingState, setButtonLoading, renderRagMarkdown, flashRagPanel, addTokens } =
    window.UIUtilities;
  const { renderInstinctAgentResult, renderPlaybookAgentResult, extractTokenCount } =
    window.AgentHandlers;

  const csvInput = document.getElementById(`enhance-${agent}-csv`);
  const notesInput = document.getElementById(`enhance-${agent}-notes`);
  const btn = document.getElementById(`enhance-${agent}-btn`);
  const output = document.getElementById(`${agent}_output`);

  const formData = new FormData();
  if (agent === 'instinct' && lastInstinctContext) {
    formData.append('previous_result', JSON.stringify(lastInstinctContext.result));
  }
  if (agent === 'playbook' && lastPlaybookContext) {
    formData.append('previous_result', JSON.stringify(lastPlaybookContext.result));
  }
  if (csvInput && csvInput.files.length > 0) {
    formData.append('csv_file', csvInput.files[0]);
  }
  if (notesInput && notesInput.value.trim()) {
    formData.append('enhance_notes', notesInput.value.trim());
  }

  fetch(`/enhance_${agent}`, {
    method: 'POST',
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      hideLoadingState(agent);
      setButtonLoading(btn, false);

      const enhancedBadge =
        '<div class="border border-info rounded p-2 mb-2 bg-light"><span class="badge bg-info text-dark">Enhanced Strategy Result</span></div>';
      if (agent === 'instinct') {
        output.innerHTML = enhancedBadge + renderInstinctAgentResult(data);
      } else {
        output.innerHTML = enhancedBadge + renderPlaybookAgentResult(data);
      }

      addTokens(extractTokenCount(data));

      if (data && data.rag_insights) {
        const ragContent = document.getElementById(`rag-${agent}-content`);
        const ragPanel = document.getElementById(`rag-${agent}-panel`);
        if (ragContent) ragContent.innerHTML = renderRagMarkdown(data.rag_insights);
        if (ragPanel) ragPanel.style.display = 'block';
        flashRagPanel(agent);
      }
    });
}

// Regression predictor handler setup
function setupRegressionPredictorHandler() {
  const regBtn = document.getElementById('run-regression-strategy-btn');
  const regResultDiv = document.getElementById('regression-strategy-result');
  const regCsvInfo = document.getElementById('regression-csv-info');
  const csvFileInput = document.getElementById('csv_file');

  if (!regBtn) return;

  // Check CSV info
  if (csvFileInput && csvFileInput.files.length > 0) {
    const file = csvFileInput.files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
      const text = e.target.result;
      const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
      const csvRows = lines.length - 1; // header
      const csvInterval = file.name.match(/(\d+m|\d+h|daily|minute|hour)/i)?.[0] || '';

      if (regCsvInfo) {
        regCsvInfo.innerHTML = `<span class='text-secondary'>📄 CSV Detected: <b>${csvRows}</b> bars loaded${csvInterval ? ` (${csvInterval})` : ''}</span>`;
        if (csvRows < 500) {
          regCsvInfo.innerHTML += ` <span class='text-warning ms-2'>⚠️ Not enough bars (min: 500 required)</span>`;
          regBtn.disabled = true;
        } else {
          regBtn.disabled = false;
        }
      }
    };
    reader.readAsText(file);
  } else if (regCsvInfo) {
    regCsvInfo.innerHTML = `<span class='text-danger'>No CSV uploaded. Please upload a CSV to run regression strategy.</span>`;
    regBtn.disabled = true;
  }
}

// Export functions
window.FormHandlers = {
  initializeFormHandlers,
  setupChatHandlers,
  setupEnhancementHandlers,
  fetchEnhance,
  setupRegressionPredictorHandler,
};
