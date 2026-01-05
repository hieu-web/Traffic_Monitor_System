# ?? Traffic Monitor System

A computer vision-based application designed to detect, track, and monitor vehicle traffic flow and detect violations in real-time.

## ?? Live Demo
Experience the application directly in your browser:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://hieu-traffic-monitor.streamlit.app/)

## ?? Features
* **Vehicle Detection**: Identify and classify cars, motorbikes, buses, and trucks.
* **Real-time Monitoring**: Process video streams for live traffic analysis.
* **Violation Detection**: Automatically capture evidence of vehicles crossing the stop line during red/yellow lights.
* **License Plate OCR**: Integrated PaddleOCR for potential plate recognition.
* **Data Analytics**: Live counting and classification statistics.

## ?? Project Structure
* `app.py`: The main Streamlit web application.
* `models/`: Contains the pre-trained YOLO model (`best.pt`).
* `requirements.txt`: Configuration for the cloud environment.

## ?? How to Use (Web Version)
1. **Access the Demo**: Click the "Streamlit App" badge above.
2. **Upload Video**: Use the sidebar to upload a traffic video file (.mp4, .avi).
3. **Configure ROI**: Adjust the Yellow box (Light ROI) to cover the traffic light in the video.
4. **Set Stop Line**: Move the slider to align the Red/Green line with the actual road stop line.
5. **View Results**: Watch real-time counting and download violation evidence if needed.

## ?? Tech Stack
* **Language**: Python
* **AI Frameworks**: Ultralytics (YOLOv8/v10), PaddleOCR.
* **UI**: Streamlit.
* **Processing**: OpenCV (Headless).

---
*Developed by Hieu-Web*