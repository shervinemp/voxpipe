"""Execution engine synchronization and runtime locks."""

import threading

# Shared lock for ONNX Runtime C++ session execution across threads (ASR, VAD, TTS, LLM)
onnx_lock = threading.Lock()

__all__ = ["onnx_lock"]
