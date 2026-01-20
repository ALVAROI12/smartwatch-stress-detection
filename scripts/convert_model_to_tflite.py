"""Shim: moved to pipelines/convert_model_to_tflite.py"""
from pipelines.convert_model_to_tflite import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("pipelines.convert_model_to_tflite", run_name="__main__")
