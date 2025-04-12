import streamlit as st
import sys
import os

# Show a startup message
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show a loading indicator
with st.spinner("Loading application dependencies..."):
    try:
        # Check for OpenCV dependencies before importing
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            from PIL import Image
            import datetime
            import tempfile
            
            # Import the remaining modules
            import os
            import io
            import time
            import json
            from pathlib import Path
            import matplotlib.pyplot as plt
            import folium
            from folium.plugins import MarkerCluster, HeatMap
            from streamlit_folium import st_folium
            import base64
            import random
            import pandas as pd
            import requests
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut
            import qrcode
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            import torch
            from ultralytics.nn.tasks import DetectionModel
            from torch.nn.modules.container import Sequential
            
            # Fix for PyTorch 2.6 weights loading issue
            torch.serialization.add_safe_globals([DetectionModel, Sequential])
            
            # If all imports successful, import the main app
            from app import *
            
        except ImportError as e:
            st.error(f"Error loading dependencies: {str(e)}")
            st.info("Try refreshing the page. If the issue persists, please contact support.")
            st.code(str(e), language="python")
            
            # Still try to show some UI even if dependencies fail
            st.title("🛣️ Pothole Detection System")
            st.markdown("""
            ## System is currently experiencing technical difficulties
            
            Our team is working to resolve this issue. The application requires specific system libraries for 
            computer vision processing.
            
            ### What you can do:
            
            1. Try refreshing the page
            2. Check back later
            3. Contact support with the error message shown above
            """)
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.code(str(e), language="python") 
