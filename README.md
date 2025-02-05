# Smart Traffic Management System

## Description:

The Smart Traffic Management System is an AI-powered solution designed to optimize traffic flow using YOLOv8 for vehicle classification and a custom algorithm to determine the optimal green light duration for traffic signals. By analyzing real-time traffic density and assigning weights to different vehicle classes, the system dynamically adjusts signal timing to reduce congestion and improve traffic efficiency.

## Features:

- **Vehicle Detection & Classification:** Uses YOLOv8 to detect and classify vehicles (two-wheelers, three-wheelers, four-wheelers, and large vehicles).
- **Weighted Traffic Analysis:** Assigns weights to vehicle types to calculate an effective traffic load.
- **Dynamic Signal Timing:** Adjusts green light duration based on real-time traffic density using a custom algorithm.
- **Optimized Traffic Flow:** Reduces congestion by prioritizing lanes with higher traffic density.
- **Scalability:** Can be integrated with existing traffic management systems.

## Run the Code:

```bash
python main.py
```

## Installation:

### Prerequisites:

- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- SQL Database (for data storage)

### Setup:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-traffic-management.git
   cd smart-traffic-management
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv8 weights:
   ```bash
   wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8n.pt
   ```

4. Run the project:
   ```bash
   python main.py
   
