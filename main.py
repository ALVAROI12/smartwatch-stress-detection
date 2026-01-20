"""Shim: moved to entrypoints/main.py"""
from entrypoints.main import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("entrypoints.main", run_name="__main__")
