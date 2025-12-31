# Surgical Visibility & Safety Monitoring System

##  Problem Statement
Minimally invasive surgeries rely on real-time camera feeds, but visibility often degrades due to smoke, bleeding, blur, or fogging. Surgeons must also control multiple instruments in a confined space under high cognitive load.
Currently, visibility loss and safety risks are detected manually, which can lead to delayed responses, instrument collisions, unsafe tool movements, increased error probability, and longer surgeries. There is no automated system that continuously monitors surgical video to assess visibility and instrument safety while providing timely, explainable alerts to assist surgeons.

## Solution Overview
We propose an **AI-assisted surgical video monitoring system** that continuously analyzes surgical footage to support surgeons during procedures.

The system:
- Monitors video quality 
- Detects visibility degradation causes (blur, smoke/fog, blood)
- Detects surgical instruments and assesses proximity-based safety risks
- Generates time-synced, explainable alerts aligned with video playback

This solution acts as a **decision-support system**, assisting surgeons without replacing human judgment.

## Core Features

###  Visibility Assessment
- Frame-wise video analysis
- Blur detection using Laplacian variance
- Blood detection using HSV-based color segmentation
- Smoke/fog estimation using contrast and intensity variation
- Visibility classified as **GOOD / MODERATE / POOR**
- Confidence score generated for each assessment

###  Instrument Safety Monitoring
- Surgical instrument detection using YOLO-based object detection
- Distance-based proximity analysis between detected tools
- Safety status flagged as **SAFE** or **RISK**

## Process Flow
Surgical Video (Uploaded)
→ Frame Extraction
→ Image Preprocessing
→ Visibility Analysis (Blur | Blood | Smoke)
→ Visibility Classification + Confidence
→ Instrument Detection (YOLO)
→ Tool Proximity Analysis
→ Alert Generation
→ Assistive Feedback on Dashboard

## Architecture Overview
User (Surgeon / Operator)
→ Web Dashboard (Frontend – Netlify)
→ Backend API (FastAPI – Render)
→ Video Analysis & Inference
→ JSON Results
→ Real-time Alerts & Visualization

## Deployment

###  Live MVP (Frontend)
https://visibility-detector.netlify.app/

### Backend API
https://visibility-detection.onrender.com/


> Backend processing is optimized for **short demo videos** due to compute limitations on free-tier hosting.

## Screenshots
## Screenshots

!(ss/Screenshot%202025-12-31%20112133.png)
!(ss/Screenshot%202025-12-31%20112142.png)
!(ss/Screenshot%202025-12-31%20112424.png)


## Technology Stack
- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI (Python)
- Computer Vision: OpenCV
- Object Detection: YOLO (Ultralytics)
- Deployment: Netlify (Frontend), Render (Backend)


## MVP Scope
- Video-based (uploaded clips, not live feed)
- Short-duration videos for demo purposes
- Real-time alert visualization during playback
- Designed for hackathon and prototype demonstration

## Disclaimer
This project is a prototype built for hackathon and research demonstration purposes only and is not intended for clinical or medical use.

## Team
Developed as a collaborative hackathon project exploring AI-assisted surgical safety and visibility enhancement.





