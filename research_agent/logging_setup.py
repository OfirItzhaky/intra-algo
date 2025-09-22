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

class SimpleFormatter(logging.Formatter):
    """Simple readable formatter for development"""
    def format(self, record):
        return f"{record.asctime} - {record.name} - {record.levelname} - {record.getMessage()}"

def setup_logging(level="INFO", use_json=None, force_setup=False):
    # Check if logging was already set up (prevent double setup)
    root = logging.getLogger()
    if root.handlers and not force_setup:
        print(f"[LOGGING] Already configured, skipping setup")
        # But still ensure Flask loggers work
        for name in ['werkzeug', 'flask', 'app', __name__]:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            logger.propagate = True
        return root
    
    # Resolve level from str or int
    lvl = logging._nameToLevel.get(level, logging.INFO) if isinstance(level, str) else int(level)

    # Auto-detect if we should use JSON (production) or simple (development)
    if use_json is None:
        use_json = os.getenv("USE_JSON_LOGGING", "false").lower() == "true"

    # Remove any existing handlers and attach ours to ROOT
    for h in list(root.handlers):
        root.removeHandler(h)

    # Create handler with unbuffered output
    h = logging.StreamHandler(stream=sys.stdout)
    h.flush = lambda: sys.stdout.flush()  # Force flush on every log
    
    # Choose formatter based on environment
    if use_json:
        h.setFormatter(JsonFormatter())
        print(f"[LOGGING] Using JSON formatter - Level: {level}")
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        h.setFormatter(formatter)
        print(f"[LOGGING] Using simple formatter - Level: {level}")
    
    root.addHandler(h)
    root.setLevel(lvl)
    
    # Ensure Flask/werkzeug logs are visible
    logging.getLogger('werkzeug').setLevel(lvl)
    logging.getLogger('flask').setLevel(lvl)

    # Make sure every existing named logger bubbles up to root
    for name, lg in logging.root.manager.loggerDict.items():
        if isinstance(lg, logging.Logger):
            lg.propagate = True
            # Don't let child loggers silently filter out INFO
            if lg.level == logging.NOTSET:
                # leave it; it will inherit root's level
                pass
            else:
                # if someone set a higher level elsewhere, drop it to root lvl
                lg.setLevel(min(lg.level, lvl))

def get_logger(name=None):
    lg = logging.getLogger(name or "app")
    lg.propagate = True
    return lg
