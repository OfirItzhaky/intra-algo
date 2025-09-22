# 🚀 Deployment Instructions for Intra-Algo App (GCP Cloud Run)

This guide outlines the full process to update, test, build, and deploy the Intra-Algo Flask app from GCP Cloud Shell. Use this whenever you push local changes to GitHub and want to deploy them.

---

## ✅ 1. Open Cloud Shell

Log into your GCP account, open Cloud Shell, and navigate to the project directory:

```bash
cd ~/intra-algo
```

## 🔄 2. Pull Latest Code from GitHub

```bash
git pull origin main
```

If you see .venv or __pycache__ issues, delete them:

```bash
rm -rf venv backend/analyzer/__pycache__ research_agent/**/__pycache__
```

## 🐍 3. Create a New Virtual Environment (Optional but recommended)

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧪 4. (Optional) Run App Locally for Testing

Set working directory:

```bash
cd ~/intra-algo
```

Set environment variables temporarily for testing:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export FINNHUB_KEY=...
# Add all your other variables here as needed
```

Run the app:

```bash
python research_agent/app.py
```

Test that the login prompt appears. ✅ The session will now expire when the browser is closed, due to this line in app.py:

```python
session.permanent = False
```

## 🐳 5. Build the Docker Image

```bash
docker build -t intra-algo-app .
```

## 📦 6. Tag and Push Image to Artifact Registry

Replace <your-region> if needed (you used europe-west4):

```bash
docker tag intra-algo-app europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app
docker push europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app
```

## ☁️ 7. Deploy to Cloud Run with Environment Variables

```bash
gcloud run deploy intra-algo-service \
  --image=europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app \
  --platform=managed \
  --region=europe-west4 \
  --allow-unauthenticated \
  --port=8080 \
  --set-env-vars="OPENAI_API_KEY=sk-...,GEMINI_API_KEY=...,FINNHUB_KEY=...,FMP_KEY=...,REDDIT_CLIENT_ID=...,REDDIT_CLIENT_SECRET=...,REDDIT_PASSWORD=...,REDDIT_USER_AGENT=...,REDDIT_USERNAME=...,ITZ_OPENAI_API_KEY=..."
```
   or if no changes to env vars can also deploy using this:
```bash
gcloud run deploy intra-algo-service \
  --image=europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app \
  --platform=managed \
  --region=europe-west4 \
  --allow-unauthenticated \
  --port=8080
```
✅ Note: You can also store env vars in `.env.yaml` and use `--env-vars-file=.env.yaml` for cleaner reuse.

## 🧪 8. Final Test (Production)

Open the Cloud Run URL in incognito or new browser.

Confirm:

- Login prompt appears ✅
- Uploads and analysis work ✅
- Session expires after browser closes ✅

## 📁 Optional Cleanup

```bash
docker system prune -a
```

✅ You're done!

Push updates from local and repeat these steps to redeploy.

---

### IF ONLY FIXING AND SHELL IS OPEN JUST DO:

1. GIT PULL  
2. `docker build -t intra-algo-app .`  
3. docker tag intra-algo-app europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app
4. docker push europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app
### make sure to tag the correct image then push the right one before deploying...

```bash
gcloud run deploy intra-algo-app \
  --image=europe-west4-docker.pkg.dev/research-agent-459210/intra-algo-repo/intra-algo-app \
  --platform=managed \
  --region=europe-west4 \
  --allow-unauthenticated

```