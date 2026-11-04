"""Sightline backend (FastAPI), built during the Meta hackathon event window.

Wraps the live pipeline (`sightline_pipeline`) for the mobile app and the dev console:
JWT auth, camera registry + one pipeline worker per active camera, streamed/chunked video
upload -> looping virtual camera, events feed with range-streamed clips, low-fps live preview,
WebSocket alerts and Expo push notifications. Run: see backend/README.md.
"""
__version__ = "0.1.0"
