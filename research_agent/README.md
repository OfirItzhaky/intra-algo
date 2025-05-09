# Research Agent

A powerful AI-powered research assistant application for analyzing market data, processing news, and interpreting chart images.

## Application Overview

This Flask-based web application provides financial market analysis tools including:

- **Daily Market Analysis**: Summarizes market news, economic events, and company insights
- **Momentum Analysis**: Calculates technical momentum indicators for stocks using SMA crossovers
- **Image Analysis**: Interprets chart images using AI vision capabilities

## Button Functionality

The application provides several key functions accessible via buttons on the main dashboard:

### Daily Analysis
This button triggers a comprehensive analysis of the current market environment including:
- Market bias determination (bullish/bearish/neutral)
- Top sector identification based on smart money flow
- Upcoming economic events
- News summaries for specific symbols
- Company profile information

### Momentum Analysis
Performs technical analysis on stock symbols:
- Analyzes both weekly and daily timeframes
- Calculates momentum using SMA (Simple Moving Average) crossovers
- Color-codes results (green for bullish, yellow for caution, red for bearish)
- Detects when price touches moving averages (potential support/resistance)
- Works with both external market data and user-uploaded CSV files

### Image Analysis
Allows chart image analysis through three methods:
- **Upload**: Upload chart images directly
- **Clipboard**: Analyze screenshots from clipboard
- **Pick Files**: Select multiple image files from your computer

### Upload CSV
Enables users to upload custom price data:
- Accepts CSV, TXT, and Excel files
- Automatically detects timeframes (daily, weekly, intraday)
- Validates data quality and structure
- Makes uploaded data available for momentum analysis

### Reset
Clears all current analysis results and uploaded files to start fresh.

## Deployment Instructions

### Prerequisites
- Google Cloud Platform account
- Google Cloud CLI installed
- Git repository for your project

### Local Development

1. **Clone the repository**:
   ```
   git clone <your-repository-url>
   cd research_agent
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run locally**:
   ```
   python app.py
   ```
   The application will be available at http://localhost:8080

### Cloud Deployment

1. **Initial Setup** (one-time only):
   ```
   # Initialize gcloud
   gcloud init

   # Configure Docker for gcloud
   gcloud auth configure-docker
   ```

2. **Deployment Process**:

   a. **Commit and push your changes**:
   ```
   git add .
   git commit -m "Your commit message"
   git push
   ```

   b. **Build the Docker image**:
   ```
   gcloud builds submit --tag gcr.io/research-agent-459210/research-agent ./research_agent/
   ```
   *Note: Run this command from the parent directory of research_agent*

   c. **Deploy to Cloud Run**:
   ```
   gcloud run deploy research-agent-service --image gcr.io/research-agent-459210/research-agent --platform managed --region us-west1
   ```

3. **Update Existing Deployment**:
   After making changes, follow the same process:
   ```
   git add .
   git commit -m "Description of changes"
   git push
   
   gcloud builds submit --tag gcr.io/research-agent-459210/research-agent ./research_agent/
   gcloud run deploy research-agent-service --image gcr.io/research-agent-459210/research-agent --platform managed --region us-west1
   ```

### Troubleshooting

- **Build Failures**: Check the build logs for specific error messages
- **Runtime Errors**: The application now displays detailed error messages instead of generic 500 errors
- **Dependencies**: Ensure all required libraries are listed in requirements.txt
- **Environment Variables**: Check that all required API keys are properly set

## Environment Configuration

The application requires several API keys for full functionality:
- OpenAI API key (for AI analysis)
- Alpha Vantage API key (for market data)
- Finnhub API key (for news and sentiment)

These should be configured in the config.py file or as environment variables in your Cloud Run deployment.

## Notes for Production

- The application has been modified to work in non-interactive environments (Cloud Run)
- Image analysis features that require GUI libraries (like tkinter) will degrade gracefully when running in the cloud
- Interactive Jupyter notebook widgets are only used when available

For additional help or questions, refer to the project documentation or contact the development team. 