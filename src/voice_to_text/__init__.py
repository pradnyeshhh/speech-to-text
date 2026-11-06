"""voice_to_text: hold a key, speak, and the text is pasted where you're typing."""

import os

# onnxruntime (it runs the Silero VAD) otherwise records device and process details for Microsoft's
# telemetry service, and its upload thread can segfault the process at exit. This has to be set
# before onnxruntime is imported. Set ORT_DISABLE_TELEMETRY=0 to allow it.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
