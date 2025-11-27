# BirdGuard — AI-Powered Tool for Legal & Ethical Hunting

**BirdGuard** is a computer‑vision system designed to assist conservation officers, environmental authorities, and automated monitoring stations in rapidly identifying **legal** versus **illegally captured** birds.  
It uses a YOLO-based model deployed on hardware such as the Raspberry Pi to provide **real-time detection** with a physical LED indicator.

---

## 📁 Repository Structure

```
BirdGuard/
├── pimodelv1.1.pt                            # Trained YOLO model (v1.1)
├── run_and_show_led_multithread_onillegal.py # Live camera detection (illegal mode)
├── run_and_show_led_multithread_onlegal.py   # Live camera detection (legal mode)
├── BirdGuard_Training_Notebook.ipynb         # Model training / dataset prep
├── BirdGuard_Image_Demo_FromRepo.ipynb       # Upload-image demo + LED emulator
└── README.md
```

---

## Project Overview

BirdGuard offers two primary ways to use the model:

### **1. Real-Time Raspberry Pi Deployment**
Using the GPIO-enabled scripts:
- Captures camera frames continuously  
- Performs YOLO detection on a background thread  
- Uses timing logic to confirm detection for **1 continuous second**  
- Activates a physical LED:
  - 🔴 Illegal → LED turns **on** (illegal mode)  
  - 🟢 Legal   → LED turns **on** (legal mode)  

### **2. Notebook Demo (No Hardware Needed)**
`BirdGuard_Image_Demo_FromRepo.ipynb`:
- Upload any bird image  
- Runs YOLO detection using the same trained weights  
- Displays a simulated LED indicator:
  - 🔴 Blinking red = illegal detected  
  - 🟢 Solid green = legal detected  
  - ⚪ Gray = no confident detection  

---

## Model Information

- **Architecture:** YOLO (Ultralytics)
- **Model file:** `pimodelv1.1.pt`
- **Class mapping used for LED logic:**
  - `class 1` → Illegal bird  
  - `class 0` → Legal bird  
- **Thresholds used in Pi deployment scripts:**
  - `CONF_TRIGGER = 0.10`
  - `IOU = 0.45`
  - `REQUIRED_ON_TIME = 1 second`

---

## 🛠 Installation & Setup

### Install dependencies
```bash
pip install ultralytics opencv-python ipywidgets matplotlib
```

### Clone the repository
```bash
git clone https://github.com/aazella/BirdGuard.git
cd BirdGuard
```

---

## Run the Demo (Image Upload)

Open:
```
BirdGuard_Image_Demo_FromRepo.ipynb
```

Then:
1. Run all cells  
2. Upload an image  
3. Click **Run BirdGuard**  
4. View LED emulator + detection summary  

No hardware required.

---

## 🟢 Raspberry Pi Deployment (Real-Time LED)

### Hardware requirements:
- Raspberry Pi  
- LED connected to GPIO pin 17  
- Camera module (PiCam or USB Camera preferably with 10x zoom)  
- `lgpio` installed  

### Illegal-mode:
```bash
python3 run_and_show_led_multithread_onillegal.py
```

### Legal‑mode:
```bash
python3 run_and_show_led_multithread_onlegal.py
```

LED will activate only when the corresponding bird class is detected **continuously for 1 second**.

---

## 🧰 Training the Model

`BirdGuard_Training_Notebook.ipynb` includes:
- Dataset preparation  
- YOLO training pipeline  
- Evaluation & export  
- Instructions to retrain on new classes  

---

## 👥 Authors

Developed by:  
**Adam Yahya • Yasmina Hanna • Nour El Semrani**  
for **EECE 490 — Supervised by Dr. Ammar Mohanna**

---


## 🙌 Acknowledgments

- Ultralytics YOLO  
- Raspberry Pi Foundation  
- iNaturalist
