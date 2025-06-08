import base64
import json
import requests
from datetime import datetime
from pathlib import Path
import imghdr  # make sure this is at the top of your file
import os

# Try to import optional dependencies, but don't fail if they're not available
try:
    import ipywidgets as widgets
    from IPython.display import display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

try:
    from PIL import ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class ImageAnalyzerAI:
    def __init__(self, model_provider, model_name, api_key):
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.image_analysis = {}

        self.prompts = [
            {
                "id": "trend_strength",
                "name": "Trend Strength Index",
                "timeframe": "weekly",
                "prompt": (
                    "Analyze the uploaded weekly chart using the following logic:\n"
                    "1. Add EMA(10) and EMA(20) to the chart.\n"
                    "2. If EMA(10) > EMA(20) AND the bar closes above EMA(10), assign +1 for positive momentum.\n"
                    "3. If the current bar OR the previous green bar touches EMA(10), assign +1. If not, assign -1.\n"
                    "4. If momentum is positive but the close is below the previous pivot and within 1.5 ATR, override score to 0.\n"
                    "Return a label (Bullish, Neutral, or Bearish), a numeric score (-1, 0, 1, or 2), and a short natural language explanation."
                )
            },
            {
                "id": "entry_timing",
                "name": "Entry Timing Index",
                "timeframe": "daily",
                "prompt": (
                    "Analyze the uploaded daily chart using the following logic:\n"
                    "1. Add EMA(10) and EMA(20).\n"
                    "2. If EMA(10) > EMA(20) AND the bar closes above EMA(10), assign +1.\n"
                    "3. If the current bar OR previous green bar touches EMA(10), assign +1. If not, assign -1.\n"
                    "4. If momentum is positive but the close is below yesterday's pivot and within 1.5 ATR, override to 0.\n"
                    "Return a label (Bullish, Neutral, or Bearish), a score, and a brief explanation."
                )
            }
        ]

    def get_prompt_by_id(self, rule_id):
        for rule in self.prompts:
            if rule["id"] == rule_id:
                return rule
        return None

    def upload_and_analyze_images(self, rule_id):
        if not JUPYTER_AVAILABLE:
            print("Jupyter widgets not available in this environment")
            return
            
        self._uploader = widgets.FileUpload(
            accept='.png,.jpg,.jpeg',
            multiple=True,
            description="Upload Snapshot Images"
        )
        display(self._uploader)
        self._uploader.observe(lambda change: self._handle_uploaded_files(rule_id), names='value')

    def _handle_uploaded_files(self, rule_id):
        prompt_block = self.get_prompt_by_id(rule_id)
        if not prompt_block:
            print(f"❌ Invalid rule_id: {rule_id}")
            return

        # Create temp directory if it doesn't exist
        temp_dir = os.path.abspath("temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        results = {}
        for filename, fileinfo in self._uploader.value.items():
            symbol = filename.split(".")[0].upper()
            image_bytes = fileinfo['content']
            print(f"🧠 Analyzing {symbol}...")

            temp_path = os.path.join(temp_dir, filename)
            try:
                with open(temp_path, "wb") as temp:
                    temp.write(image_bytes)

                response = self.analyze_image_with_bytes(temp_path, rule_id)

                if isinstance(response, dict) and "label" in response.get("raw_output", "").lower():
                    results[symbol] = {
                        "raw_output": response["raw_output"],
                        "timeframe": prompt_block["timeframe"]
                    }
                else:
                    results[symbol] = {
                        "label": "Unknown",
                        "score": 0,
                        "timeframe": prompt_block["timeframe"],
                        "explanation": f"⚠️ No valid pattern detected. Please upload the appropriate image for {symbol}."
                    }

            except Exception as e:
                results[symbol] = {
                    "label": "Error",
                    "score": 0,
                    "timeframe": prompt_block["timeframe"],
                    "explanation": f"⚠️ Failed to process image for {symbol}: {e}"
                }
            finally:
                # Clean up the temporary file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {filename}: {e}")

        self.image_analysis = results
        print("✅ Image analysis completed.")

    def analyze_image_with_bytes(self, image_input, rule_id):
        """
        Analyze an image from either a file path or raw bytes.
        """
        prompt_block = self.get_prompt_by_id(rule_id)
        if not prompt_block:
            raise ValueError(f"Unknown rule_id: {rule_id}")

        # 🔍 Check if input is bytes or file path
        if isinstance(image_input, bytes):
            image_bytes = image_input
        elif isinstance(image_input, str):  # assume it's a path
            with open(image_input, "rb") as image_file:
                image_bytes = image_file.read()
        else:
            raise ValueError("Invalid input: must be file path or raw bytes")

        if self.model_provider == "openai":
            return self._analyze_with_openai(image_bytes, prompt_block["prompt"])
        elif self.model_provider == "gemini":
            return self._analyze_with_gemini(image_bytes, prompt_block["prompt"])
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _analyze_with_openai(self, image_bytes, prompt):
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Proper MIME detection
        mime_type = imghdr.what(None, h=image_bytes) or "png"
        image_data = f"data:image/{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": image_data
                    }}
                ]}
            ],
            "max_tokens": 500
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return {"raw_output": content}

    def _analyze_with_gemini(self, image_bytes, prompt):
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        mime_type = imghdr.what(None, h=image_bytes) or "image/png"
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": f"image/{mime_type}",
                                "data": encoded_image
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(endpoint, headers=headers, json=body)
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return {"raw_output": content}

    def pick_snapshots_and_analyze(self, rule_id="trend_strength"):
        """
        Opens a file dialog to pick image files and analyzes them.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            file_paths = filedialog.askopenfilenames(
                title="Select snapshot images",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
            )
            
            if not file_paths:
                print("No files selected.")
                return {}

            # No need to create temporary files since we're using the original files directly
            results = {}
            for path in file_paths:
                symbol = Path(path).stem.upper()
                print(f"🧠 Analyzing {symbol}...")

                try:
                    # Using the file directly rather than creating a temporary copy
                    result = self.analyze_image_with_bytes(path, rule_id)
                    results[symbol] = {
                        "raw_output": result["raw_output"],
                        "timeframe": self.get_prompt_by_id(rule_id)["timeframe"]
                    }
                except Exception as e:
                    results[symbol] = {
                        "label": "Error",
                        "score": 0,
                        "timeframe": self.get_prompt_by_id(rule_id)["timeframe"],
                        "explanation": f"⚠️ Failed to process image: {e}"
                    }

            self.image_analysis = results
            print("✅ All snapshots processed.")
            return results
        
        except ImportError:
            print("Tkinter not available in this environment")
            return {}
        except Exception as e:
            print(f"❌ Error in file dialog: {e}")
            return {}

    def analyze_clipboard_snapshot(self, rule_id="trend_strength"):
        """
        Analyzes an image from the clipboard.
        """
        if not PIL_AVAILABLE:
            print("PIL ImageGrab not available in this environment")
            return
            
        print("📋 Waiting for snapshot from clipboard...")

        image = ImageGrab.grabclipboard()
        if image is None:
            print("❌ No image found in clipboard. Use Snipping Tool or press PrtScr, then try again.")
            return

        # Create a temp folder if it doesn't exist
        temp_dir = os.path.abspath("temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the file to the temp directory
        file_name = f"snapshot_{datetime.now().strftime('%H%M%S')}.png"
        image_path = os.path.join(temp_dir, file_name)
        image.save(image_path)

        try:
            print(f"✅ Snapshot saved as: {os.path.basename(image_path)}")
            result = self.analyze_image_with_bytes(image_path, rule_id)
            print("🔍 Analysis Result:")
            print(result["raw_output"])
            return result
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
                    print(f"✓ Removed temporary file: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {e}") 