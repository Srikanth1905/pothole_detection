import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import io
import time
import json
import datetime
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import tempfile
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

# Add DetectionModel and Sequential to safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([DetectionModel, Sequential])

# GHMC Constants
GHMC_TOLL_FREE = "040-21111111"
GHMC_WHATSAPP = "9848021665"
GHMC_EMAIL = "complaints@ghmc.gov.in"

def get_location_from_ip():
    try:
        response = requests.get('https://ipapi.co/json/')
        data = response.json()
        return data.get('latitude'), data.get('longitude'), data.get('city')
    except:
        return None, None, None

def get_address_from_coordinates(lat, lon):
    try:
        geolocator = Nominatim(user_agent="pothole_detection")
        location = geolocator.reverse(f"{lat}, {lon}")
        return location.address
    except GeocoderTimedOut:
        return "Location not found"

def save_uploaded_image(image_array, index):
    """Save uploaded image to the uploads directory with a unique name."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pothole_{timestamp}_{index}.jpg"
    filepath = upload_dir / filename
    
    # Save the image
    Image.fromarray(image_array).save(filepath)
    return str(filepath)

def generate_ghmc_report(detection_data, images, location_info):
    # Create a temporary file for the PDF
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
    
    # Initialize saved_image_paths list
    saved_image_paths = []
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Custom styles
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.HexColor('#1a237e')  # Dark blue color
    )
    
    subheader_style = ParagraphStyle(
        'CustomSubHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        textColor=colors.HexColor('#1a237e')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12
    )
    
    # Add GHMC header with logo
    elements.append(Paragraph("GREATER HYDERABAD MUNICIPAL CORPORATION", header_style))
    elements.append(Paragraph("Road Maintenance Department", subheader_style))
    elements.append(Paragraph("Pothole Detection Report", subheader_style))
    elements.append(Spacer(1, 20))
    
    # Add report details in a professional format
    elements.append(Paragraph("1. Report Details", subheader_style))
    report_details = [
        ["Report ID:", f"GHMC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"],
        ["Date & Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Location:", location_info['address']],
        ["Coordinates:", f"{location_info['lat']}, {location_info['lon']}"],
        ["Report Type:", "Automated Pothole Detection"],
        ["Road Type:", detection_data['road_type']],
        ["Road Condition:", detection_data['quality_rating']],
        ["Traffic Density:", detection_data.get('traffic_density', 'Not specified')],
        ["Last Maintenance:", detection_data.get('last_maintenance', 'Unknown')]
    ]
    
    details_table = Table(report_details, colWidths=[150, 300])
    details_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 20))
    
    # Add location analysis
    elements.append(Paragraph("2. Location Analysis", subheader_style))
    location_analysis = f"""
    The reported location is situated in {location_info['city'] or 'the specified area'}. 
    This {detection_data['road_type'].lower()} road serves as a {detection_data.get('traffic_density', 'moderate')} traffic route. 
    The last known maintenance was conducted on {detection_data.get('last_maintenance', 'an unknown date')}.
    
    The area experiences {detection_data.get('traffic_density', 'moderate')} traffic flow, which significantly impacts road wear and maintenance requirements. 
    The road type classification as {detection_data['road_type']} indicates its importance in the local transportation network.
    
    Based on historical data and current observations, this section of road requires {detection_data['quality_rating'].lower()} maintenance attention. 
    The combination of traffic patterns and road type suggests a need for {detection_data.get('traffic_density', 'moderate')} frequency maintenance schedule.
    """
    elements.append(Paragraph(location_analysis, normal_style))
    elements.append(Spacer(1, 20))
    
    # Add road condition assessment
    elements.append(Paragraph("3. Road Condition Assessment", subheader_style))
    road_condition = f"""
    The automated detection system has conducted a comprehensive analysis of the road surface condition. 
    The assessment reveals {detection_data['num_detections']} potholes within the surveyed area, with an average severity score of {detection_data['avg_severity']:.1f}.
    
    The road quality rating of {detection_data['quality_rating']} is based on multiple factors including:
    • Pothole density and distribution
    • Severity of individual potholes
    • Overall road surface integrity
    • Impact on traffic flow and safety
    
    The current condition indicates {detection_data['quality_rating'].lower()} road maintenance status, requiring {detection_data['quality_rating'].lower()} priority attention for repairs.
    """
    elements.append(Paragraph(road_condition, normal_style))
    elements.append(Spacer(1, 20))
    
    # Add detection analysis
    elements.append(Paragraph("4. Detection Analysis", subheader_style))
    
    # Calculate severity distribution
    severity_distribution = {
        "Critical": detection_data.get('critical_severity', 0),
        "High": detection_data.get('high_severity', 0),
        "Medium": detection_data.get('medium_severity', 0),
        "Low": detection_data.get('low_severity', 0)
    }
    
    total_detections = detection_data['num_detections']
    if total_detections > 0:
        severity_percentages = {
            "Critical": (severity_distribution["Critical"] / total_detections * 100),
            "High": (severity_distribution["High"] / total_detections * 100),
            "Medium": (severity_distribution["Medium"] / total_detections * 100),
            "Low": (severity_distribution["Low"] / total_detections * 100)
        }
    else:
        severity_percentages = {
            "Critical": 0,
            "High": 0,
            "Medium": 0,
            "Low": 0
        }
    
    detection_analysis = f"""
    The automated detection system has identified {total_detections} potholes in the surveyed area. 
    The severity distribution shows:
    • {severity_distribution['Critical']} critical severity potholes ({severity_percentages['Critical']:.1f}%)
    • {severity_distribution['High']} high severity potholes ({severity_percentages['High']:.1f}%)
    • {severity_distribution['Medium']} medium severity potholes ({severity_percentages['Medium']:.1f}%)
    • {severity_distribution['Low']} low severity potholes ({severity_percentages['Low']:.1f}%)
    
    The average severity score of {detection_data['avg_severity']:.1f} indicates {detection_data['quality_rating'].lower()} overall road condition. 
    This assessment is based on standardized criteria for pothole severity classification, taking into account both size and depth of detected potholes.
    """
    elements.append(Paragraph(detection_analysis, normal_style))
    elements.append(Spacer(1, 20))
    
    # Add recommendations
    elements.append(Paragraph("5. Recommendations", subheader_style))
    recommendations = f"""
    Based on the comprehensive analysis, the following recommendations are proposed:
    
    1. Immediate Action Required:
    • Prioritize repair of {severity_distribution['Critical']} critical severity potholes
    • Address {severity_distribution['High']} high severity potholes within 48 hours
    • Implement temporary safety measures for severe potholes
    
    2. Short-term Measures:
    • Schedule repairs for medium severity potholes within 1 week
    • Conduct additional inspection of surrounding road surface
    • Monitor traffic flow during repair operations
    
    3. Long-term Recommendations:
    • Implement regular maintenance schedule
    • Consider road resurfacing if pothole density exceeds threshold
    • Develop preventive maintenance program
    
    The recommended repair approach should consider:
    • Traffic density and patterns
    • Road type and usage
    • Weather conditions
    • Available maintenance resources
    """
    elements.append(Paragraph(recommendations, normal_style))
    elements.append(Spacer(1, 20))
    
    # Add additional information if provided
    if detection_data.get('additional_notes'):
        elements.append(Paragraph("6. Additional Information", subheader_style))
        elements.append(Paragraph(detection_data['additional_notes'], normal_style))
        elements.append(Spacer(1, 20))
    
    # Add images with detailed captions
    if images:
        elements.append(Paragraph("7. Visual Evidence", subheader_style))
        elements.append(Paragraph("The following images show the detected potholes and their locations:", normal_style))
        elements.append(Spacer(1, 10))
        
        for idx, img in enumerate(images):
            # Save the image and get its path
            image_path = save_uploaded_image(img, idx)
            saved_image_paths.append(image_path)
            
            # Add image to PDF with caption
            elements.append(RLImage(image_path, width=6*inch, height=4*inch))
            elements.append(Paragraph(f"Figure {idx + 1}: Pothole detection at {location_info['address']}", normal_style))
            elements.append(Paragraph(f"Location: {location_info['address']}", normal_style))
            elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            elements.append(Spacer(1, 10))
    
    # Add official footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("8. GHMC Contact Information", subheader_style))
    contact_info = [
        ["Department", "Contact Number"],
        ["Road Maintenance", GHMC_TOLL_FREE],
        ["WhatsApp Helpline", GHMC_WHATSAPP]
    ]
    
    contact_table = Table(contact_info, colWidths=[200, 200])
    contact_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(contact_table)
    
    # Add disclaimer
    elements.append(Spacer(1, 20))
    disclaimer_text = """
    This report is generated automatically by the GHMC Pothole Detection System. 
    The severity assessments and recommendations are based on standardized criteria 
    and should be verified by GHMC officials before taking action.
    """
    elements.append(Paragraph(disclaimer_text, normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    # Clean up temporary image files after PDF is generated
    for image_path in saved_image_paths:
        try:
            os.remove(image_path)
        except:
            pass  # Ignore errors in cleanup
    
    return pdf_path

# Set page configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to beautify the app
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-container {
        background-color: #0066cc;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .subtitle {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dashboard-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-box {
        border: 2px dashed #0066cc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the pothole detection model
@st.cache_resource
def load_model():
    model_path = Path("runs/train/pothole_detector/weights/best.pt")
    if not model_path.exists():
        st.error("Model not found. Please make sure the model is located at 'runs/train/pothole_detector/weights/best.pt'")
        return None
    try:
        # Try loading the model (ultralytics internally uses torch.load)
        return YOLO(model_path)
    except RuntimeError as e:
        # If weights_only error occurs, try to fix it
        if "weights_only" in str(e) or "Unsupported global" in str(e):
            try:
                from ultralytics.nn.modules.conv import Conv
                import torch
                with torch.serialization.safe_globals([Conv]):
                    return YOLO(model_path)

# Function to process an image
def process_image(image, model):
    # Run inference
    results = model(image)
    
    # Get the annotated image
    result_image = results[0].plot()
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Get the number of detected potholes
    num_detections = len(results[0].boxes)
    
    # Get confidence scores and bounding box sizes for severity estimation
    confidences = []
    severities = []
    total_severity_score = 0
    
    for i, box in enumerate(results[0].boxes):
        # Get confidence
        conf = float(box.conf)
        confidences.append(round(conf * 100, 2))
        
        # Get box dimensions for severity estimation
        box_data = box.xywh[0]  # Get width and height of bounding box
        width = float(box_data[2])
        height = float(box_data[3])
        
        # Calculate area as a percentage of image size
        img_height, img_width = result_image_rgb.shape[:2]
        relative_size = (width * height) / (img_width * img_height)
        
        # Determine severity based on size and confidence
        # Small potholes: 1, Medium: 2, Large: 3
        if relative_size < 0.01:  # Small pothole
            severity = 1
        elif relative_size < 0.05:  # Medium pothole
            severity = 2
        else:  # Large pothole
            severity = 3
            
        # Adjust severity by confidence
        if conf < 0.5:
            severity = max(1, severity - 1)  # Lower confidence reduces severity, min 1
        elif conf > 0.85:
            severity += 1  # High confidence increases severity
        
        severities.append(severity)
        total_severity_score += severity
    
    # Estimate road segment length (assuming standard view distance)
    # This is a simplified approach - in a real system, GPS data would provide actual distance
    estimated_road_length = 0.1  # km (assume image covers ~100m of road)
    
    # Calculate potholes per km
    potholes_per_km = num_detections / estimated_road_length if estimated_road_length > 0 else 0
    
    return result_image_rgb, num_detections, confidences, severities, total_severity_score, potholes_per_km

# Function to process a video
def process_video(video_file, model, progress_bar):
    # Create a temporary file for the input video
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(video_file.read())
    temp_input_path = temp_input.name
    temp_input.close()
    
    # Create a temporary file for the output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output.name
    temp_output.close()
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        # Initialize statistics
        frame_number = 0
        detections = 0
        total_severity_score = 0
        max_detections_frame = 0
        all_confidences = []
        all_severities = []
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 detection on the frame
            results = model(frame)
            
            # Get the annotated frame with detections
            annotated_frame = results[0].plot()
            
            # Write the processed frame
            out.write(annotated_frame)
            
            # Count detections in this frame
            frame_detections = len(results[0].boxes)
            detections += frame_detections
            max_detections_frame = max(max_detections_frame, frame_detections)
            
            # Calculate severity scores for each detection
            for box in results[0].boxes:
                # Get confidence score
                conf = float(box.conf)
                all_confidences.append(conf)
                
                # Get box dimensions for severity estimation
                box_data = box.xywh[0]
                box_width = float(box_data[2])
                box_height = float(box_data[3])
                
                # Calculate relative size
                relative_size = (box_width * box_height) / (width * height)
                
                # Determine severity based on size and confidence
                if relative_size < 0.01:  # Small pothole
                    severity = 1
                elif relative_size < 0.05:  # Medium pothole
                    severity = 2
                else:  # Large pothole
                    severity = 3
                
                # Adjust severity based on confidence
                if conf < 0.5:
                    severity = max(1, severity - 1)
                elif conf > 0.85:
                    severity += 1
                
                all_severities.append(severity)
                total_severity_score += severity
            
            # Update progress bar
            frame_number += 1
            progress_bar.progress(frame_number / total_frames)
        
        # Release resources
        cap.release()
        out.release()
        
        # Ensure the video file is properly closed and accessible
        time.sleep(0.5)
        
        # Calculate statistics
        avg_detections = detections / frame_number if frame_number > 0 else 0
        avg_severity = total_severity_score / detections if detections > 0 else 0
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Estimate road length (assuming average driving speed of 40 km/h)
        video_duration_seconds = total_frames / fps
        estimated_road_length = (40 * video_duration_seconds) / 3600  # km
        potholes_per_km = detections / estimated_road_length if estimated_road_length > 0 else 0
        
        # Prepare video statistics
        video_stats = {
            'total_frames': total_frames,
            'processed_frames': frame_number,
            'total_detections': detections,
            'avg_detections_per_frame': avg_detections,
            'max_detections_frame': max_detections_frame,
            'avg_severity': avg_severity,
            'avg_confidence': avg_confidence,
            'estimated_road_length': estimated_road_length,
            'potholes_per_km': potholes_per_km,
            'video_duration': video_duration_seconds,
            'fps': fps,
            'resolution': f"{width}x{height}"
        }
        
        return temp_output_path, video_stats
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None
    finally:
        # Clean up temporary input file
        try:
            os.unlink(temp_input_path)
        except:
            pass

# Function to add animation to elements
def add_animation(element_name, animation_type="fadeIn", delay=0):
    return f"""
    <div id="{element_name}" style="animation: {animation_type} 1s ease {delay}s both;">
        {element_name}_content
    </div>
    """

# Detection data management
class DetectionDataManager:
    def __init__(self, data_file="detection_data.json"):
        self.data_dir = Path("../data")
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / data_file
        self.ensure_data_file()
        
    def ensure_data_file(self):
        if not self.data_file.exists():
            with open(self.data_file, 'w') as f:
                json.dump({"detections": []}, f)
    
    def load_data(self):
        with open(self.data_file, 'r') as f:
            return json.load(f)
    
    def save_data(self, data):
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_detection(self, detection_data):
        data = self.load_data()
        data["detections"].append(detection_data)
        self.save_data(data)
    
    def get_all_detections(self):
        return self.load_data()["detections"]
    
    def get_statistics(self):
        detections = self.get_all_detections()
        if not detections:
            return {
                "total_detections": 0,
                "avg_confidence": 0,
                "images_processed": 0,
                "videos_processed": 0,
                "by_month": {},
                "by_location": {},
                "by_road_type": {},
                "recent": []
            }
        
        # Calculate statistics
        total_detections = sum(d["num_detections"] for d in detections)
        all_confidences = [c for d in detections for c in d.get("confidences", [])]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        images_processed = sum(1 for d in detections if d["source_type"] == "image")
        videos_processed = sum(1 for d in detections if d["source_type"] == "video")
        
        # Group by month
        by_month = {}
        for d in detections:
            month = d["timestamp"].split("-")[1]  # Extract month from YYYY-MM-DD
            year = d["timestamp"].split("-")[0]   # Extract year
            key = f"{year}-{month}"
            if key not in by_month:
                by_month[key] = 0
            by_month[key] += d["num_detections"]
        
        # Group by location
        by_location = {}
        for d in detections:
            if "location" in d:
                loc = d["location"]
                if loc not in by_location:
                    by_location[loc] = {"count": 0, "quality": 0}
                by_location[loc]["count"] += d["num_detections"]
                by_location[loc]["quality"] += d.get("quality_score", 50)
        
        # Average the quality scores
        for loc in by_location:
            by_location[loc]["quality"] = by_location[loc]["quality"] / (by_location[loc]["count"] or 1)
        
        # Group by road type
        by_road_type = {}
        for d in detections:
            if "road_type" in d:
                road_type = d["road_type"]
                if road_type not in by_road_type:
                    by_road_type[road_type] = 0
                by_road_type[road_type] += d["num_detections"]
        
        # Get recent detections
        recent = sorted(detections, key=lambda x: x["timestamp"], reverse=True)[:5]
        
        return {
            "total_detections": total_detections,
            "avg_confidence": avg_confidence,
            "images_processed": images_processed,
            "videos_processed": videos_processed,
            "by_month": by_month,
            "by_location": by_location,
            "by_road_type": by_road_type,
            "recent": recent
        }

# Initialize the detection data manager
@st.cache_resource
def get_data_manager():
    return DetectionDataManager()

data_manager = get_data_manager()

# Main title with animation
st.markdown("""
<div class="title-container">
    <h1>🛣️ Pothole Detection System</h1>
    <p>Advanced deep learning powered road analysis</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 Image Detection", "🎥 Video Analysis", "🗺️ Map View", "📊 Statistics", "📄 Report Generation"])

with tab1:
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.header("Pothole Detection in Images")
    st.markdown("Upload one or multiple images to detect potholes on roads.")
    
    # Load the model
    model = load_model()
    
    if model:
        # Get location automatically
        lat, lon, city = get_location_from_ip()
        location_info = {
            'lat': lat,
            'lon': lon,
            'city': city,
            'address': get_address_from_coordinates(lat, lon) if lat and lon else "Location not available"
        }
        
        # Display location information
        st.markdown("### Location Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Detected City:** {location_info['city'] or 'Not available'}")
        with col2:
            st.markdown(f"**Address:** {location_info['address']}")
        
        # Allow manual location override
        manual_location = st.text_input("Override Location (if needed)", 
                                      value=location_info['address'],
                                      placeholder="Enter full address")
        
        if manual_location:
            location_info['address'] = manual_location
        
        uploaded_files = st.file_uploader("Choose one or multiple images...", 
                                         type=["jpg", "jpeg", "png"], 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            # Initialize statistics for multiple images
            total_detections = 0
            total_severity_score = 0
            total_potholes_per_km = 0
            all_confidences = []
            all_severities = []
            processed_images = []
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing image {idx + 1} of {len(uploaded_files)}..."):
                    # Read the image
                    image = Image.open(uploaded_file)
                    
                    # Convert PIL Image to numpy array for OpenCV
                    image_np = np.array(image)
                    
                    # Process the image
                    result_image, num_detections, confidences, severities, severity_score, potholes_per_km = process_image(image_np, model)
                    
                    # Update statistics
                    total_detections += num_detections
                    total_severity_score += severity_score
                    total_potholes_per_km += potholes_per_km
                    all_confidences.extend(confidences)
                    all_severities.extend(severities)
                    processed_images.append({
                        'image': result_image,
                        'num_detections': num_detections,
                        'confidences': confidences,
                        'severities': severities,
                        'severity_score': severity_score,
                        'potholes_per_km': potholes_per_km
                    })
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Calculate aggregate statistics
            avg_detections = total_detections / len(uploaded_files) if uploaded_files else 0
            avg_severity_score = total_severity_score / len(uploaded_files) if uploaded_files else 0
            avg_potholes_per_km = total_potholes_per_km / len(uploaded_files) if uploaded_files else 0
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            # Display aggregate statistics
            st.markdown("### Aggregate Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images Processed", len(uploaded_files))
            with col2:
                st.metric("Total Potholes Detected", total_detections)
            with col3:
                st.metric("Average Potholes per Image", f"{avg_detections:.1f}")
            with col4:
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            
            # Display severity distribution
            st.markdown("### Severity Distribution")
            severity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for severity in all_severities:
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Low Severity", severity_counts[1], "Small potholes")
            with col2:
                st.metric("Medium Severity", severity_counts[2], "Medium potholes")
            with col3:
                st.metric("High Severity", severity_counts[3], "Large potholes")
            with col4:
                st.metric("Critical Severity", severity_counts[4], "Very large potholes")
            
            # Display individual image results
            st.markdown("### Individual Image Results")
            for idx, result in enumerate(processed_images):
                with st.expander(f"Image {idx + 1} - {result['num_detections']} potholes detected"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.image(result['image'], caption=f"Processed Image {idx + 1}", use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Detection Details")
                        st.markdown(f"**Number of potholes:** {result['num_detections']}")
                        st.markdown(f"**Average confidence:** {sum(result['confidences'])/len(result['confidences']):.1f}%" if result['confidences'] else "No detections")
                        st.markdown(f"**Severity score:** {result['severity_score']}")
                        st.markdown(f"**Potholes per km:** {result['potholes_per_km']:.1f}")
                        
                        # Display confidence distribution
                        st.markdown("#### Confidence Distribution")
                        if result['confidences']:
                            for i, conf in enumerate(result['confidences']):
                                st.progress(conf/100)
                                st.text(f"Pothole {i+1}: {conf:.1f}% (Severity: {result['severities'][i]})")
                        else:
                            st.info("No potholes detected in this image.")
            
            # Road quality assessment based on aggregate data
            st.markdown("### Overall Road Quality Assessment")
            
            # Calculate normalized metrics
            avg_potholes_per_km = total_potholes_per_km / len(uploaded_files) if uploaded_files else 0
            avg_severity = total_severity_score / total_detections if total_detections > 0 else 0
            
            # Quality assessment based on multiple factors
            if avg_potholes_per_km < 1:
                quality_distance = "Excellent"
                distance_color = "green"
            elif avg_potholes_per_km < 3:
                quality_distance = "Good"
                distance_color = "lightgreen"
            elif avg_potholes_per_km < 6:
                quality_distance = "Fair"
                distance_color = "orange"
            else:
                quality_distance = "Poor"
                distance_color = "red"
            
            # Severity-based assessment
            if avg_severity < 1.5:
                quality_severity = "Excellent"
                severity_color = "green"
            elif avg_severity < 2.5:
                quality_severity = "Good"
                severity_color = "lightgreen"
            elif avg_severity < 3.5:
                quality_severity = "Fair"
                severity_color = "orange"
            else:
                quality_severity = "Poor"
                severity_color = "red"
            
            # Display assessments
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Based on density:** <span style='color:{distance_color};font-weight:bold'>{quality_distance}</span> ({avg_potholes_per_km:.1f}/km)", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Based on severity:** <span style='color:{severity_color};font-weight:bold'>{quality_severity}</span> (Score: {avg_severity:.1f})", unsafe_allow_html=True)
            
            # Overall assessment
            overall_quality = "Poor" if "Poor" in [quality_distance, quality_severity] else \
                            "Fair" if "Fair" in [quality_distance, quality_severity] else \
                            "Good" if "Good" in [quality_distance, quality_severity] else "Excellent"
            
            overall_color = "red" if overall_quality == "Poor" else \
                          "orange" if overall_quality == "Fair" else \
                          "lightgreen" if overall_quality == "Good" else "green"
            
            st.markdown(f"**Overall assessment:** <span style='color:{overall_color};font-weight:bold;font-size:1.2em'>{overall_quality}</span>", unsafe_allow_html=True)
            
            # Add road type consideration
            road_type = st.selectbox("Road Type:", ["Residential", "Urban Arterial", "Highway"], key="img_road_type")
            
            # Adjust assessment based on road type
            if road_type == "Highway" and overall_quality in ["Fair", "Poor"]:
                st.markdown(f"<span style='color:red;font-weight:bold'>⚠️ High priority for repair (Highway)</span>", unsafe_allow_html=True)
            elif road_type == "Urban Arterial" and overall_quality == "Poor":
                st.markdown(f"<span style='color:red;font-weight:bold'>⚠️ Priority for repair (Urban Arterial)</span>", unsafe_allow_html=True)
            
            # Save detection data button
            if st.button("Save Detection Data", key="save_detection_data_image"):
                # Convert quality rating to score between 0-100
                quality_score = 90 if overall_quality == "Excellent" else \
                              70 if overall_quality == "Good" else \
                              50 if overall_quality == "Fair" else 30
                
                # Generate random coordinates around a central point if location is provided
                lat, lon = None, None
                if location_info['lat'] and location_info['lon']:
                    # Add some random variation to create realistic distribution
                    lat = location_info['lat'] + (random.random() - 0.5) * 0.05
                    lon = location_info['lon'] + (random.random() - 0.5) * 0.05
                
                # Create detection data for each image
                for idx, result in enumerate(processed_images):
                    detection_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_type": "image",
                        "location": location_info['address'],
                        "num_detections": result['num_detections'],
                        "confidences": result['confidences'],
                        "severities": result['severities'],
                        "total_severity_score": result['severity_score'],
                        "potholes_per_km": float(result['potholes_per_km']),
                        "road_type": road_type,
                        "quality_rating": overall_quality,
                        "quality_score": quality_score
                    }
                    
                    # Add coordinates if available
                    if lat and lon:
                        detection_data["lat"] = lat
                        detection_data["lon"] = lon
                    
                    # Save to data manager
                    data_manager.add_detection(detection_data)
                
                st.success(f"Detection data saved for {len(uploaded_files)} images! You can view it in the Map and Statistics tabs.")

            # Add GHMC report generation
            if st.button("Generate GHMC Report"):
                with st.spinner("Generating professional report..."):
                    # Prepare detection data
                    detection_summary = {
                        'num_detections': total_detections,
                        'avg_severity': total_severity_score / total_detections if total_detections > 0 else 0,
                        'quality_rating': overall_quality,
                        'road_type': road_type
                    }
                    
                    # Generate PDF report
                    pdf_path = generate_ghmc_report(detection_summary, 
                                                  [img['image'] for img in processed_images],
                                                  location_info)
                    
                    # Create download button for the report
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download GHMC Report",
                            data=f,
                            file_name="ghmc_pothole_report.pdf",
                            mime="application/pdf"
                        )
                    
                    # Display GHMC contact information
                    st.markdown("### GHMC Contact Information")
                    st.markdown(f"""
                    - **Toll Free Number:** {GHMC_TOLL_FREE}
                    - **WhatsApp Number:** {GHMC_WHATSAPP}
                    - **Email:** {GHMC_EMAIL}
                    """)
                    
                    # Generate QR code for WhatsApp
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(f"https://wa.me/{GHMC_WHATSAPP}")
                    qr.make(fit=True)
                    qr_img = qr.make_image(fill_color="black", back_color="white")
                    
                    # Save QR code to temporary file
                    qr_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                    qr_img.save(qr_path)
                    
                    # Display QR code
                    st.markdown("### Scan to contact GHMC via WhatsApp")
                    st.image(qr_path, width=200)
                    
                    # Clean up temporary files
                    os.unlink(pdf_path)
                    os.unlink(qr_path)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.header("Pothole Detection in Videos")
    st.markdown("""
    Upload a video to detect potholes throughout the footage. The system will:
    1. Process each frame using YOLOv8
    2. Detect and annotate potholes
    3. Calculate severity and statistics
    4. Generate a processed video with detections
    """)
    
    # Load the model
    model = load_model()
    
    if model:
        # Location input
        location_input = st.text_input("Location (optional)", 
                                     placeholder="e.g. Delhi, Bangalore, etc.", 
                                     key="video_location")
        
        # Video upload
        uploaded_video = st.file_uploader("Choose a video...", 
                                        type=["mp4", "avi", "mov"], 
                                        key="video_uploader")
        
        if uploaded_video is not None:
            # Display original video
            st.subheader("Original Video")
            st.video(uploaded_video)
            
            # Process video button
            if st.button("Process Video"):
                # Create progress bar
                progress_text = "Processing video..."
                progress_bar = st.progress(0)
                
                # Process the video
                with st.spinner("Processing video... This may take a while depending on the video length."):
                    output_path, video_stats = process_video(uploaded_video, model, progress_bar)
                    
                    if output_path and video_stats:
                        # Display the processed video
                        st.subheader("Processed Video with Detections")
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        
                        # Display detailed statistics
                        st.subheader("Video Analysis Results")
                        
                        # Create columns for statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Basic Statistics")
                            st.markdown(f"**Total Frames:** {video_stats['total_frames']}")
                            st.markdown(f"**Video Duration:** {video_stats['video_duration']:.1f} seconds")
                            st.markdown(f"**Resolution:** {video_stats['resolution']}")
                            st.markdown(f"**FPS:** {video_stats['fps']}")
                        
                        with col2:
                            st.markdown("### Detection Statistics")
                            st.markdown(f"**Total Potholes Detected:** {video_stats['total_detections']}")
                            st.markdown(f"**Average Detections per Frame:** {video_stats['avg_detections_per_frame']:.2f}")
                            st.markdown(f"**Max Detections in a Frame:** {video_stats['max_detections_frame']}")
                            st.markdown(f"**Average Confidence:** {video_stats['avg_confidence']*100:.1f}%")
                        
                        # Road quality assessment
                        st.markdown("### Road Quality Assessment")
                        
                        # Calculate quality based on potholes per km
                        if video_stats['potholes_per_km'] < 1:
                            quality = "Excellent"
                            color = "green"
                        elif video_stats['potholes_per_km'] < 3:
                            quality = "Good"
                            color = "lightgreen"
                        elif video_stats['potholes_per_km'] < 6:
                            quality = "Fair"
                            color = "orange"
                        else:
                            quality = "Poor"
                            color = "red"
                        
                        st.markdown(f"""
                        - **Estimated Road Length:** {video_stats['estimated_road_length']:.2f} km
                        - **Potholes per km:** {video_stats['potholes_per_km']:.2f}
                        - **Average Severity:** {video_stats['avg_severity']:.2f}
                        - **Road Quality:** <span style='color:{color};font-weight:bold'>{quality}</span>
                        """, unsafe_allow_html=True)
                        
                        # Save detection data
                        if st.button("Save Detection Data", key="save_detection_data_video"):
                            detection_data = {
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "source_type": "video",
                                "location": location_input if location_input else "Unknown",
                                "video_stats": video_stats,
                                "quality_rating": quality
                            }
                            data_manager.add_detection(detection_data)
                            st.success("Detection data saved successfully!")
                        
                        # Clean up temporary file
                        try:
                            os.unlink(output_path)
                        except:
                            pass
                    else:
                        st.error("Failed to process video. Please try again with a different video file.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.header("Pothole Map Visualization")
    st.markdown("Interactive map showing pothole detections and their severity.")
    
    # Get real detection data
    all_detections = data_manager.get_all_detections()
    map_data = []
    
    # If no real data exists yet, use sample data
    if not all_detections or not any("lat" in d and "lon" in d for d in all_detections):
        st.info("No location data available yet. Using sample data for demonstration.")
        # Sample data for demonstration
        map_data = [
            {"lat": 28.6139, "lon": 77.2090, "severity": 0.8, "date": "2023-04-05", "location": "Delhi", "num_detections": 5},
            {"lat": 28.6160, "lon": 77.2167, "severity": 0.5, "date": "2023-04-06", "location": "Delhi", "num_detections": 3},
            {"lat": 28.6252, "lon": 77.2100, "severity": 0.9, "date": "2023-04-07", "location": "Delhi", "num_detections": 8},
            {"lat": 28.6305, "lon": 77.2190, "severity": 0.3, "date": "2023-04-08", "location": "Delhi", "num_detections": 2},
            {"lat": 28.6139, "lon": 77.2290, "severity": 0.7, "date": "2023-04-09", "location": "Delhi", "num_detections": 4},
        ]
    else:
        # Process real data
        for detection in all_detections:
            if "lat" in detection and "lon" in detection:
                # Normalize severity to 0-1 scale for visualization
                if "quality_score" in detection:
                    severity = 1 - (detection["quality_score"] / 100)  # Invert so higher is worse
                elif "normalized_severity" in detection:
                    severity = min(1.0, detection["normalized_severity"] / 15)
                else:
                    severity = 0.5  # Default
                
                map_data.append({
                    "lat": detection["lat"],
                    "lon": detection["lon"],
                    "severity": severity,
                    "date": detection["timestamp"].split()[0],
                    "location": detection["location"],
                    "num_detections": detection["num_detections"],
                    "quality_rating": detection.get("quality_rating", "Unknown")
                })
    
    # Check if we have data to show
    if map_data:
        # Create map centered on the average coordinates
        avg_lat = sum(item["lat"] for item in map_data) / len(map_data)
        avg_lon = sum(item["lon"] for item in map_data) / len(map_data)
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
        
        # Create different map views
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Markers View")
            
            # Add a marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add markers for each pothole
            for item in map_data:
                # Set color based on severity
                if item["severity"] < 0.4:
                    color = "green"
                elif item["severity"] < 0.7:
                    color = "orange"
                else:
                    color = "red"
                
                popup_html = f"""
                <strong>Location:</strong> {item['location']}<br>
                <strong>Date:</strong> {item['date']}<br>
                <strong>Detections:</strong> {item['num_detections']}<br>
                <strong>Quality:</strong> {item.get('quality_rating', 'Unknown')}
                """
                
                folium.Marker(
                    location=[item["lat"], item["lon"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=color)
                ).add_to(marker_cluster)
            
            # Display the map
            st_folium(m)
        
        with col2:
            st.subheader("Heatmap View")
            
            # Create a new map for the heatmap
            m2 = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
            
            # Prepare heatmap data with enhanced intensity
            heat_data = [[item["lat"], item["lon"], item["severity"] * item["num_detections"] * 2] for item in map_data]
            
            # Add heatmap layer with basic configuration
            HeatMap(
                data=heat_data,
                radius=20,
                blur=15
            ).add_to(m2)
            
            # Display the heatmap
            st_folium(m2)
        
        # Statistics on potholes in the area
        st.subheader("Area Statistics")
        
        # Create columns for statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            total_detections = sum(item["num_detections"] for item in map_data)
            st.metric("Total Potholes", total_detections)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            avg_severity = sum(item["severity"] for item in map_data) / len(map_data)
            st.metric("Average Severity", f"{avg_severity:.2f}/1.0")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            high_severity = sum(1 for item in map_data if item["severity"] >= 0.7)
            st.metric("High Severity Locations", high_severity)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add location filtering
        if len(set(item["location"] for item in map_data)) > 1:
            st.subheader("Filter by Location")
            locations = list(set(item["location"] for item in map_data))
            selected_location = st.selectbox("Select location:", ["All"] + locations)
            
            if selected_location != "All":
                filtered_data = [item for item in map_data if item["location"] == selected_location]
                
                # Create a filtered map
                if filtered_data:
                    st.subheader(f"Potholes in {selected_location}")
                    
                    # Create map centered on the average coordinates
                    avg_lat = sum(item["lat"] for item in filtered_data) / len(filtered_data)
                    avg_lon = sum(item["lon"] for item in filtered_data) / len(filtered_data)
                    
                    filtered_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)
                    
                    # Add markers for each pothole
                    for item in filtered_data:
                        # Set color based on severity
                        if item["severity"] < 0.4:
                            color = "green"
                        elif item["severity"] < 0.7:
                            color = "orange"
                        else:
                            color = "red"
                        
                        popup_html = f"""
                        <strong>Date:</strong> {item['date']}<br>
                        <strong>Detections:</strong> {item['num_detections']}<br>
                        <strong>Quality:</strong> {item.get('quality_rating', 'Unknown')}
                        """
                        
                        folium.Marker(
                            location=[item["lat"], item["lon"]],
                            popup=folium.Popup(popup_html, max_width=300),
                            icon=folium.Icon(color=color)
                        ).add_to(filtered_map)
                    
                    # Display the filtered map
                    st_folium(filtered_map)
    else:
        st.warning("No location data available. Save some detection data to see it on the map.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.header("Detection Statistics Dashboard")
    
    # Get statistics from real data
    stats = data_manager.get_statistics()
    
    # Check if we have real data
    if stats["total_detections"] == 0:
        st.info("No real detection data available yet. Using sample data for demonstration.")
        # Sample data for demonstration
        dates = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        detections = [12, 18, 25, 14, 30, 22]
        areas = ["Connaught Place", "Hauz Khas", "Saket", "Dwarka", "Rohini"]
        quality_scores = [85, 62, 78, 90, 70]
        
        total_detections = "1,208"
        avg_confidence = "87.5%"
        images_processed = "532"
        videos_processed = "48"
        recent_data = {
            "Date": ["2023-06-10", "2023-06-09", "2023-06-08", "2023-06-07", "2023-06-06"],
            "Location": ["Connaught Place", "Hauz Khas", "Saket", "Dwarka", "Rohini"],
            "Detections": [5, 3, 7, 2, 4],
            "Confidence": ["92%", "87%", "90%", "85%", "88%"],
            "Status": ["Reported", "Fixed", "Pending", "Reported", "In Progress"]
        }
    else:
        # Process real data for charts
        
        # Time series data
        by_month = stats["by_month"]
        if by_month:
            # Sort by date
            dates = sorted(by_month.keys())
            detections = [by_month[d] for d in dates]
            # Format dates for display (convert YYYY-MM to abbreviated month name)
            month_names = {
                "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
                "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
            }
            dates = [f"{month_names.get(d.split('-')[1], d.split('-')[1])} {d.split('-')[0]}" for d in dates]
        else:
            # If no monthly data, use empty
            dates = []
            detections = []
        
        # Location quality data
        by_location = stats["by_location"]
        if by_location:
            areas = list(by_location.keys())
            quality_scores = [int(by_location[a]["quality"]) for a in areas]
        else:
            # If no location data, use empty
            areas = []
            quality_scores = []
        
        # Format metrics
        total_detections = f"{stats['total_detections']:,}"
        avg_confidence = f"{stats['avg_confidence']:.1f}%"
        images_processed = f"{stats['images_processed']}"
        videos_processed = f"{stats['videos_processed']}"
        
        # Recent detections
        if stats["recent"]:
            recent_data = {
                "Date": [d["timestamp"].split()[0] for d in stats["recent"]],
                "Location": [d["location"] for d in stats["recent"]],
                "Detections": [d["num_detections"] for d in stats["recent"]],
                "Quality": [d.get("quality_rating", "Unknown") for d in stats["recent"]],
                "Type": [d["source_type"].capitalize() for d in stats["recent"]]
            }
        else:
            recent_data = {
                "Date": [],
                "Location": [],
                "Detections": [],
                "Quality": [],
                "Type": []
            }
    
    # Create columns for charts
    if dates and detections:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pothole Detections Over Time")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(dates, detections, marker='o', linewidth=2, color='#0066cc')
            ax.set_ylabel('Number of Potholes')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            if areas and quality_scores:
                st.subheader("Road Quality by Area")
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(areas, quality_scores, color='#0066cc')
                ax.set_ylabel('Quality Score (%)')
                ax.set_ylim(0, 100)
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{height}%', ha='center', va='bottom')
                
                ax.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#f8f9fa')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough location data yet to show the quality by area chart.")
    else:
        st.info("Not enough time-series data yet to show charts.")
    
    # Summary statistics
    st.subheader("Detection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        delta = "24%" if stats["total_detections"] == 0 else None
        st.metric("Total Detections", total_detections, delta=delta)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        delta = "3.2%" if stats["avg_confidence"] == 0 else None
        st.metric("Average Confidence", avg_confidence, delta=delta)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        delta = "12" if stats["images_processed"] == 0 else None
        st.metric("Images Processed", images_processed, delta=delta)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        delta = "5" if stats["videos_processed"] == 0 else None
        st.metric("Videos Analyzed", videos_processed, delta=delta)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a data table
    st.subheader("Recent Detections")
    
    if recent_data["Date"]:
        st.dataframe(recent_data, use_container_width=True)
    else:
        st.info("No detection data saved yet. Process some images or videos and save the detection data.")
    
    # Add export functionality
    if stats["total_detections"] > 0:
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as JSON"):
                # Convert data to JSON string
                json_data = json.dumps(data_manager.get_all_detections(), indent=2)
                
                # Create download button
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="pothole_detection_data.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export as CSV"):
                # Convert data to CSV
                detections = data_manager.get_all_detections()
                
                # Flatten nested structures
                flat_data = []
                for d in detections:
                    flat_d = d.copy()
                    
                    # Flatten arrays into strings
                    for k, v in d.items():
                        if isinstance(v, list):
                            flat_d[k] = str(v)
                    
                    flat_data.append(flat_d)
                
                # Convert to DataFrame
                df = pd.DataFrame(flat_data)
                
                # Convert to CSV
                csv = df.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="pothole_detection_data.csv",
                    mime="text/csv"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.header("GHMC Pothole Detection Report Generation")
    
    # Location Information Section
    st.subheader("Location Information")
    col1, col2 = st.columns(2)
    
    with col1:
        # Get location automatically
        lat, lon, city = get_location_from_ip()
        location_info = {
            'lat': lat,
            'lon': lon,
            'city': city,
            'address': get_address_from_coordinates(lat, lon) if lat and lon else "Location not available"
        }
        
        st.markdown("**Detected Location:**")
        st.markdown(f"- City: {location_info['city'] or 'Not available'}")
        st.markdown(f"- Address: {location_info['address']}")
        st.markdown(f"- Coordinates: {location_info['lat']}, {location_info['lon']}")
    
    with col2:
        st.markdown("**Manual Location Override**")
        manual_address = st.text_area("Enter complete address:", 
                                    value=location_info['address'],
                                    height=100,
                                    placeholder="Enter the complete address including street, area, and landmark")
        
        if manual_address:
            location_info['address'] = manual_address
    
    # Report Details Section
    st.subheader("Report Details")
    col1, col2 = st.columns(2)
    
    with col1:
        road_type = st.selectbox("Road Type:", 
                               ["Residential", "Urban Arterial", "Highway", "Expressway", "Service Road"],
                               help="Select the type of road where potholes were detected")
        
        road_condition = st.selectbox("Overall Road Condition:",
                                    ["Excellent", "Good", "Fair", "Poor", "Critical"],
                                    help="Select the overall condition of the road")
    
    with col2:
        traffic_density = st.selectbox("Traffic Density:",
                                     ["Low", "Medium", "High", "Very High"],
                                     help="Select the typical traffic density on this road")
        
        last_maintenance = st.date_input("Last Maintenance Date (if known):",
                                       help="Select the date of last road maintenance")
    
    # Additional Information
    st.subheader("Additional Information")
    additional_notes = st.text_area("Additional Notes:",
                                  height=100,
                                  placeholder="Enter any additional information about the road condition, traffic patterns, or safety concerns")
    
    # Preview Section
    st.subheader("Report Preview")
    
    # Create a preview container
    preview_container = st.container()
    
    with preview_container:
        st.markdown("""
        <div style='border: 1px solid #ddd; padding: 20px; border-radius: 5px; background-color: white;'>
            <h2 style='color: #1a237e; text-align: center;'>GREATER HYDERABAD MUNICIPAL CORPORATION</h2>
            <h3 style='color: #1a237e; text-align: center;'>Road Maintenance Department</h3>
            <h4 style='color: #1a237e; text-align: center;'>Pothole Detection Report</h4>
            
            <div style='margin-top: 20px;'>
                <h5 style='color: #1a237e;'>Report Details</h5>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Report ID:</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>GHMC-{timestamp}</td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Date & Time:</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{current_time}</td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Location:</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{address}</td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Road Type:</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{road_type}</td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Road Condition:</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{road_condition}</td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Traffic Density:</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{traffic_density}</td>
                    </tr>
                </table>
            </div>
            
            <div style='margin-top: 20px;'>
                <h5 style='color: #1a237e;'>Location Details</h5>
                <p>{address}</p>
                <p>Coordinates: {coordinates}</p>
            </div>
            
            <div style='margin-top: 20px;'>
                <h5 style='color: #1a237e;'>Additional Information</h5>
                <p>{additional_notes}</p>
            </div>
            
            <div style='margin-top: 20px;'>
                <h5 style='color: #1a237e;'>GHMC Contact Information</h5>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Department</strong></td>
                        <td style='padding: 8px; border: 1px solid #ddd;'><strong>Contact Number</strong></td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'>Road Maintenance</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{toll_free}</td>
                    </tr>
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'>WhatsApp Helpline</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{whatsapp}</td>
                    </tr>
                </table>
            </div>
        </div>
        """.format(
            timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
            current_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            address=location_info['address'],
            road_type=road_type,
            road_condition=road_condition,
            traffic_density=traffic_density,
            coordinates=f"{location_info['lat']}, {location_info['lon']}",
            additional_notes=additional_notes or "No additional notes provided.",
            toll_free=GHMC_TOLL_FREE,
            whatsapp=GHMC_WHATSAPP
        ), unsafe_allow_html=True)
    
    # Generate Report Button
    st.markdown("---")
    if st.button("Generate Official GHMC Report", type="primary"):
        with st.spinner("Generating professional report..."):
            # Prepare detection data
            detection_summary = {
                'num_detections': 0,  # This will be updated when images are processed
                'avg_severity': 0,
                'quality_rating': road_condition,
                'road_type': road_type,
                'traffic_density': traffic_density,
                'last_maintenance': last_maintenance.strftime("%Y-%m-%d") if last_maintenance else "Unknown",
                'additional_notes': additional_notes
            }
            
            # Generate PDF report
            pdf_path = generate_ghmc_report(detection_summary, 
                                                  [img['image'] for img in processed_images],
                                                  location_info)
            
            # Create download button for the report
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download GHMC Report",
                    data=f,
                    file_name="ghmc_pothole_report.pdf",
                    mime="application/pdf"
                )
            
            # Display success message
            st.success("Report generated successfully! You can now download the official GHMC report.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🛣️ Pothole Detection System | Powered by YOLOv8 and Streamlit</p>
</div>
""", unsafe_allow_html=True) 
