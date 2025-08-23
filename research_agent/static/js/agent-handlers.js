/**
 * Agent Handlers Module
 * Functions for rendering and handling different agent results
 */

// Helper: Render InstinctAgent output in a structured way
function renderInstinctAgentResult(data) {
  if (!data || typeof data !== 'object') {
    return '<div class="alert alert-warning">No result.</div>';
  }
  let html = '';
  // Feedback (if present)
  if (data.feedback) {
    html += `<div class="alert alert-warning mb-3"><strong>Feedback:</strong> <br>${data.feedback}</div>`;
  }
  // Summary
  if (data.summary) {
    html += `<div class="mb-3"><strong>💬 Summary:</strong><br><p>${data.summary}</p></div>`;
  }
  // Strategies (single or list)
  let strategies = data.strategies;
  if (strategies && !Array.isArray(strategies)) strategies = [strategies];
  if (strategies && Array.isArray(strategies) && strategies.length > 0) {
    html += `<div class="mb-3"><strong>⚙️ Strategy:</strong>`;
    strategies.forEach((strat, idx) => {
      if (!strat) return;
      html += `<div class="card mb-2"><div class="card-body p-2">`;
      html += `<table class="table table-sm mb-1"><tbody>`;
      if (strat.entry_rule) html += `<tr><th>Entry Rule</th><td>${strat.entry_rule}</td></tr>`;
      if (strat.confirmation)
        html += `<tr><th>Confirmation</th><td>${strat.confirmation}</td></tr>`;
      if (strat.stop_rule) html += `<tr><th>Stop Rule</th><td>${strat.stop_rule}</td></tr>`;
      if (strat.target_rule) html += `<tr><th>Target Rule</th><td>${strat.target_rule}</td></tr>`;
      if (strat.tags)
        html += `<tr><th>Tags</th><td>${Array.isArray(strat.tags) ? strat.tags.join(', ') : strat.tags}</td></tr>`;
      if (strat.complexity) html += `<tr><th>Complexity</th><td>${strat.complexity}</td></tr>`;
      html += `</tbody></table>`;
      html += `</div></div>`;
    });
    html += `</div>`;
  }
  // Support/Resistance Zones (if present)
  if (
    data['support_resistance_zones'] &&
    Array.isArray(data['support_resistance_zones']) &&
    data['support_resistance_zones'].length > 0
  ) {
    html += `<div class="mb-3"><strong>🧱 Support/Resistance Zones:</strong><ul>`;
    data['support_resistance_zones'].forEach(zone => {
      html += `<li>${zone}</li>`;
    });
    html += `</ul></div>`;
  }
  return html;
}

// Helper: Render PlaybookAgent output in a structured way (similar to InstinctAgent)
function renderPlaybookAgentResult(data) {
  if (!data || typeof data !== 'object') {
    return '<div class="alert alert-warning">No result.</div>';
  }
  let html = '';
  // Show input validation error if present
  if (data.feedback && data.step === 'validation_failed') {
    html += `<div class="alert alert-danger mb-3"><strong>⚠️ Input Validation Error</strong><br>${data.feedback}</div>`;
    return html; // Only show this error, skip rest
  }
  // --- Bias Summary block (multi-image) ---
  if (Array.isArray(data.bias_summary) && data.bias_summary.length > 0) {
    html += `<div class="card border-warning mb-3" style="max-width: 28rem;">
            <div class="card-header bg-warning text-dark"><strong>Bias Summary</strong></div>
            <div class="card-body pb-2 pt-2">
              <ul class="list-unstyled mb-0">`;
    data.bias_summary.forEach(bias => {
      let icon = '';
      const tf = (bias.interval || '').toLowerCase();
      if (tf.includes('day')) icon = '🗓';
      else if (tf.includes('60') || tf.includes('1h')) icon = '🕐';
      else if (tf.includes('15')) icon = '🕒';
      else if (tf.includes('30')) icon = '🕧';
      else icon = '⏱';
      const conf =
        bias.confidence !== undefined
          ? ` (Confidence: ${(bias.confidence * 100).toFixed(0)}%)`
          : '';
      html += `<li class="mb-1">${icon} <strong>${bias.interval || ''}</strong> → <span class="fw-bold">${bias.bias_direction}</span>${conf}</li>`;
    });
    html += `</ul></div></div>`;
  }
  // --- End Bias Summary block ---
  // Show Multi-Timeframe Bias Summary if present
  if (
    data.multi_tf_bias &&
    typeof data.multi_tf_bias === 'object' &&
    Object.keys(data.multi_tf_bias).length > 0
  ) {
    html += `<div class="card border-info mb-3" style="max-width: 28rem;">`;
    html += `<div class="card-header bg-info text-white"><strong>Multi-Timeframe Bias</strong></div>`;
    html += `<div class="card-body">`;
    html += `<ul class="list-unstyled mb-0">`;
    for (const [tf, bias] of Object.entries(data.multi_tf_bias)) {
      let icon = '';
      if (tf.toLowerCase().includes('day')) icon = '🗓';
      else if (tf.includes('60') || tf.includes('1h')) icon = '🕐';
      else if (tf.includes('15')) icon = '🕒';
      else if (tf.includes('30')) icon = '🕧';
      else icon = '⏱';
      html += `<li class="mb-1">${icon} <strong>${tf}:</strong> ${bias}</li>`;
    }
    html += `</ul></div></div>`;
  }
  // Show feedback if present (other cases)
  if (data.feedback) {
    html += `<div class="alert alert-warning mb-3"><strong>⚠️ ${data.feedback}</strong></div>`;
  }
  // Summary
  if (data.summary) {
    html += `<div class="mb-3"><strong>💬 Summary:</strong><br><p>${data.summary}</p></div>`;
  }
  // Strategies (single or list)
  let strategies = data.strategies;
  if (strategies && !Array.isArray(strategies)) strategies = [strategies];
  if (strategies && Array.isArray(strategies) && strategies.length > 0) {
    html += `<div class="mb-3"><strong>⚙️ Strategy:</strong>`;
    strategies.forEach((strat, idx) => {
      if (!strat) return;
      html += `<div class="card mb-2"><div class="card-body p-2">`;
      html += `<table class="table table-sm mb-1"><tbody>`;
      if (strat.entry_rule) html += `<tr><th>Entry Rule</th><td>${strat.entry_rule}</td></tr>`;
      if (strat.confirmation)
        html += `<tr><th>Confirmation</th><td>${strat.confirmation}</td></tr>`;
      if (strat.stop_rule) html += `<tr><th>Stop Rule</th><td>${strat.stop_rule}</td></tr>`;
      if (strat.target_rule) html += `<tr><th>Target Rule</th><td>${strat.target_rule}</td></tr>`;
      if (strat.tags)
        html += `<tr><th>Tags</th><td>${Array.isArray(strat.tags) ? strat.tags.join(', ') : strat.tags}</td></tr>`;
      if (strat.complexity) html += `<tr><th>Complexity</th><td>${strat.complexity}</td></tr>`;
      html += `</tbody></table>`;
      html += `</div></div>`;
    });
    html += `</div>`;
  }
  // Support/Resistance Zones (if present)
  if (
    data['support_resistance_zones'] &&
    Array.isArray(data['support_resistance_zones']) &&
    data['support_resistance_zones'].length > 0
  ) {
    html += `<div class="mb-3"><strong>🧱 Support/Resistance Zones:</strong><ul>`;
    data['support_resistance_zones'].forEach(zone => {
      html += `<li>${zone}</li>`;
    });
    html += `</ul></div>`;
  }
  return html;
}

// Try to extract token usage from response (if present)
function extractTokenCount(data) {
  if (data) {
    if (data.totalTokenCount) {
      return parseInt(data.totalTokenCount);
    }
    if (data.tokens_used) {
      return parseInt(data.tokens_used);
    }
  }
  return 800;
}

// Export functions for use in other modules
window.AgentHandlers = {
  renderInstinctAgentResult,
  renderPlaybookAgentResult,
  extractTokenCount,
};
