/**
 * Agent Configuration
 * Constants and configuration objects for the scalping agent
 */

// API Endpoints
const API_ENDPOINTS = {
  START_INSTINCT: '/start_instinct',
  START_PLAYBOOK: '/start_playbook',
  QUERY_INSTINCT: '/query_instinct',
  QUERY_PLAYBOOK: '/query_playbook',
  ENHANCE_INSTINCT: '/enhance_instinct',
  ENHANCE_PLAYBOOK: '/enhance_playbook',
  RUN_REGRESSION: '/run_regression_predictor',
  RUN_MULTITIMEFRAME: '/run_multitimeframe_agent',
  RUN_VWAP: '/run_vwap_agent',
  REGRESSION_DEFAULTS: '/regression_strategy_defaults',
  VWAP_OPTIMIZATION: '/vwap_optimization_html',
};

// Timeframe options for chart analysis
const TIMEFRAME_OPTIONS = ['15m', '30m', '60m', '4h', 'Daily', 'Weekly'];

// LLM Models available
const LLM_MODELS = {
  GPT4O: 'gpt-4o',
  GEMINI: 'gemini-1.5-pro-latest',
};

// Token pricing (example rates)
const TOKEN_RATES = {
  DEFAULT: 0.0015 / 1000, // $0.0015 per 1K tokens
};

// File type restrictions
const SUPPORTED_FILE_TYPES = {
  IMAGES: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  CSV: ['.csv', '.txt', '.xlsx', '.xls'],
};

// UI Configuration
const UI_CONFIG = {
  MAX_IMAGES_VWAP: 4,
  MAX_OPTIMIZATION_FILES: 10,
  MIN_CSV_ROWS: 500,
  LOADING_DELAY: 100,
  FLASH_DURATION: 700,
  HOVER_DELAY: 100,
};

// Error Messages
const ERROR_MESSAGES = {
  NO_IMAGES: 'Please upload or paste at least one chart image.',
  NO_CSV: 'No CSV uploaded. Please upload a CSV to run regression strategy.',
  INSUFFICIENT_DATA: 'Not enough bars (min: 500 required)',
  MAX_FILES_EXCEEDED: 'Maximum 10 files allowed.',
  FAILED_TO_LOAD: 'Failed to load defaults.',
};

// Export configuration
window.AgentConfig = {
  API_ENDPOINTS,
  TIMEFRAME_OPTIONS,
  LLM_MODELS,
  TOKEN_RATES,
  SUPPORTED_FILE_TYPES,
  UI_CONFIG,
  ERROR_MESSAGES,
};
