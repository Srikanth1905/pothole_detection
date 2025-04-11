# Pothole Detection System

A deep learning-powered road analysis system that detects and analyzes potholes using YOLOv8.

## Features

- 📸 Image detection of potholes
- 🎥 Video analysis capability
- 🗺️ Interactive map visualization
- 📊 Statistics dashboard for data analysis
- 📄 Official-looking GHMC report generation

## Deployment on Streamlit Cloud

### 1. Create a GitHub Repository

- Push all files in this directory to a new GitHub repository
- Make sure the directory structure is preserved:
  ```
  app.py
  requirements.txt
  runs/train/pothole_detector/weights/best.pt
  .streamlit/config.toml
  data/uploads/  # (empty directory)
  ```

### 2. Deploy on Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your GitHub repository
4. Set the main file path to `app.py`
5. Click "Deploy"

### Notes

- The app requires an internet connection for geolocation services
- YOLOv8 model loading may take a few seconds on initial startup
- Uploaded images are temporarily stored in the `data/uploads` directory 