# logging_setup.py
import logging, json, sys, os

class JsonFormatter(logging.Formatter):
    def format(self, r):
        payload = {
            "severity": r.levelname,
            "message": r.getMessage(),
            "logger": r.name,
            "service": os.getenv("SERVICE_NAME", "intra-algo-service"),
        }
        return json.dumps(payload)
def setup_logging(level="INFO"):
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = []
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
def get_logger(name=None):
    return logging.getLogger(name or "app")
