HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
  <title>Snapshot Trade Analyzer</title>
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
  <!-- jQuery -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <!-- Bootstrap JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <style>
    body {
      background-color: #f8f9fa;
      font-family: 'Segoe UI', Arial, sans-serif;
      padding: 20px;
    }
    .card {
      margin-bottom: 20px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      border: none;
    }
    .card-header {
      background-color: #f8f9fa;
      font-weight: 600;
      display: flex;
      align-items: center;
    }
    .card-header i {
      margin-right: 10px;
      font-size: 1.2em;
    }
    .custom-file-upload {
      border: 2px dashed #ccc;
      padding: 30px;
      text-align: center;
      cursor: pointer;
      border-radius: 5px;
      background-color: #f8f9fa;
      transition: all 0.3s;
    }
    .custom-file-upload:hover {
      border-color: #6c757d;
      background-color: #eee;
    }
    .result-pre {
      background-color: #f8f9fa;
      padding: 15px;
      border-radius: 5px;
      max-height: 300px;
      overflow-y: auto;
    }
    .results-section {
      display: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1 class="mb-4">
      <span class="text-primary me-2">📸</span>
      Snapshot Trade Analyzer
    </h1>

    <!-- File Upload Card -->
    <div class="card">
      <div class="card-header">
        <i class="text-primary">📂</i> Data Upload
      </div>
      <div class="card-body">
        <form id="upload-form" action="/upload_csv" method="POST" enctype="multipart/form-data">
          <div class="custom-file-upload mb-3" id="file-drop-area">
            <i class="text-primary fs-1">📁</i>
            <p class="mb-1">Drag and drop data files here</p>
            <p class="text-muted">(CSV, TXT, Excel formats supported)</p>
            <input type="file" id="file-input" name="csv_files" multiple accept="*" class="d-none">
            <button type="button" class="btn btn-outline-primary" id="file-select-button">Select Files</button>
          </div>
          
          <!-- File Preview Table -->
          <div id="file-preview" class="mt-3" style="display: none;">
            <h5>Selected Files</h5>
            <table class="table table-bordered table-striped">
              <thead class="table-light">
                <tr>
                  <th>File Name</th>
                  <th>Size</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody id="preview-table-body">
                <!-- Preview rows go here -->
              </tbody>
            </table>
            <button type="submit" class="btn btn-primary mt-2">Upload Files</button>
          </div>
          
          <!-- Results Table -->
          {% if file_info %}
          <div class="mt-4">
            <h5>Processed Files</h5>
            <table class="table table-bordered table-striped">
              <thead class="table-light">
                <tr>
                  <th>Symbol</th>
                  <th>Interval</th>
                  <th>Last Bar Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {% for file in file_info %}
                <tr>
                  <td>{{ file.symbol }}</td>
                  <td>{{ file.interval }}</td>
                  <td>{{ file.last_bar_date }}</td>
                  <td>
                    {% if file.status == "valid" %}
                      <span class="badge bg-success">
                        {% if "Daily" in file.interval %}
                          Valid Daily Data
                        {% elif "Weekly" in file.interval %}
                          Valid Weekly Data
                        {% elif "Monthly" in file.interval %}
                          Valid Monthly Data
                        {% elif "h" in file.interval %}
                          Valid {{ file.interval }} Intraday
                        {% elif "m" in file.interval %}
                          Valid {{ file.interval }} Data
                        {% else %}
                          Valid
                        {% endif %}
                      </span>
                    {% elif file.status == "warning" %}
                      <span class="badge bg-warning text-dark" data-bs-toggle="tooltip" title="{{ file.message }}">
                        Warning <i class="bi bi-info-circle"></i>
                      </span>
                    {% elif file.status == "error" %}
                      <span class="badge bg-danger" data-bs-toggle="tooltip" title="{{ file.message }}">
                        Error <i class="bi bi-info-circle"></i>
                      </span>
                    {% else %}
                      <span class="badge bg-secondary">Unknown</span>
                    {% endif %}
                    {% if file.message %}
                      <div class="small text-muted mt-1">{{ file.message }}</div>
                    {% endif %}
                  </td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          {% endif %}
          
          <!-- Error Message -->
          {% if upload_error %}
          <div class="alert alert-danger mt-3">
            {{ upload_error }}
          </div>
          {% endif %}
        </form>
      </div>
    </div>
    
    <!-- Action Buttons Card -->
    <div class="card">
      <div class="card-header">
        <i class="text-primary">🔍</i> Analysis Tools
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-4 mb-2">
            <form method="POST" action="/daily_analysis">
              <button type="submit" class="btn btn-primary w-100">
                <i class="me-2">🧠</i> Run Daily Market Summary
              </button>
            </form>
          </div>
          <div class="col-md-4 mb-2">
            <form method="POST" action="/momentum_analysis" target="_blank">
              <button type="submit" class="btn btn-primary w-100">
                <i class="me-2">📈</i> Generate Momentum Report
              </button>
            </form>
          </div>
          <div class="col-md-4 mb-2">
            <form method="POST" action="/reset">
              <button type="submit" class="btn btn-secondary w-100">
                <i class="me-2">🔁</i> Reset
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
    <!-- Scalping Agent Button Row -->
    <div class="row mb-4">
      <div class="col-12">
        <a href="{{ scalp_agent_url }}" class="btn btn-success btn-lg w-100">
          <i class="me-2">📈</i> Open Scalping Agent
        </a>
      </div>
    </div>
    
    <!-- Symbol Chart Card -->
    <div class="card mb-4">
      <div class="card-header">
        <h5 class="mb-0">📊 Symbol Charts</h5>
      </div>
      <div class="card-body">
        <select id="symbolSelect" class="form-select">
          {% if file_info and file_info|length > 0 %}
            {% for file in file_info %}
              {% set symbol = file.symbol.split('.')[0] %}
              <option value="{{ symbol }}">{{ symbol }}</option>
            {% endfor %}
          {% else %}
            <option value="">No data files uploaded</option>
          {% endif %}
        </select>
        <button class="btn btn-primary mt-2" onclick="window.open('/symbol_chart?symbol=' + document.getElementById('symbolSelect').value)">
          🔍 View Chart
        </button>
      </div>
    </div>
    
    <!-- Results Section -->
    <div class="row mt-4">
      <!-- Daily Market Output -->
      <div class="col-md-6">
        {% if daily_outputs %}
        <div class="card">
          <div class="card-header">
            <i class="text-primary">📊</i> Daily Market Outputs
          </div>
          <div class="card-body">
            {% for result in daily_outputs %}
            <h5>{{ result.filename }}</h5>
            <pre class="result-pre">{{ result.text }}</pre>
            {% endfor %}
          </div>
        </div>
        {% endif %}
      </div>
      
      <!-- Snapshot Output -->
      <div class="col-md-6">
        {% if image_outputs %}
        <div class="card">
          <div class="card-header">
            <i class="text-primary">📸</i> Snapshot Outputs
          </div>
          <div class="card-body">
            {% for result in image_outputs %}
            <h5>{{ result.filename }}</h5>
            <pre class="result-pre">{{ result.text }}</pre>
            <!-- HIGHLIGHTED: Cost info display -->
            {% if result.cost_info %}
            <div class="alert alert-info p-2 mt-2 mb-0 small" style="border-radius:6px;">
              💰 Estimated Gemini Cost: ${{ '%.4f' % result.cost_info.estimated_cost_usd }}
              ({{ result.cost_info.prompt_tokens }} input, {{ result.cost_info.output_tokens }} output tokens)
            </div>
            {% endif %}
            <!-- END HIGHLIGHTED -->
            {% endfor %}
          </div>
        </div>
        {% endif %}
      </div>
    </div>
  </div>

  <script>
    $(document).ready(function() {
      // Initialize tooltips
      var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
      var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
      });
      
      // File input handling
      var fileInput = $('#file-input');
      var dropArea = $('#file-drop-area');
      var fileSelectButton = $('#file-select-button');
      var filePreview = $('#file-preview');
      var previewTableBody = $('#preview-table-body');
      
      fileSelectButton.click(function() {
        fileInput.click();
      });
      
      fileInput.change(function() {
        handleFiles(this.files);
      });
      
      dropArea.on('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).addClass('border-primary');
      });
      
      dropArea.on('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('border-primary');
      });
      
      dropArea.on('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('border-primary');
        
        let dt = e.originalEvent.dataTransfer;
        if (dt.files.length) {
          handleFiles(dt.files);
        }
      });
      
      function handleFiles(files) {
        if (files.length === 0) return;
        
        // Clear preview table
        previewTableBody.empty();
        
        // Show preview
        filePreview.show();
        
        // Add files to table
        Array.from(files).forEach(file => {
          let row = $('<tr>');
          row.append($('<td>').text(file.name));
          row.append($('<td>').text(formatFileSize(file.size)));
          row.append($('<td>').text(file.type || 'text/csv'));
          previewTableBody.append(row);
        });
      }
      
      function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
        else return (bytes / 1048576).toFixed(2) + ' MB';
      }
    });
  </script>
</body>
</html>
''' 