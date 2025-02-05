Smart Traffic Management System

Introduction  
The Smart Traffic Management System is an AI-powered solution designed to optimize traffic flow using YOLOv8 for vehicle classification and a custom algorithm to determine the optimal green light duration for traffic signals. By analyzing real-time traffic density and assigning weights to different vehicle classes, the system dynamically adjusts signal timing to reduce congestion and improve traffic efficiency.

Features  
- Vehicle Detection & Classification: Uses YOLOv8 to detect and classify vehicles (two-wheelers, three-wheelers, four-wheelers, and large vehicles).  
- Weighted Traffic Analysis: Assigns weights to vehicle types to calculate an effective traffic load.  
- Dynamic Signal Timing: Adjusts green light duration based on real-time traffic density using a custom algorithm.  
- Optimized Traffic Flow: Reduces congestion by prioritizing lanes with higher traffic density.  
- Scalability: Can be integrated with existing traffic management systems.  

Installation  

Prerequisites  
- Python 3.x  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- SQL Database (for data storage)  

Setup  
1. Clone the repository:  
   git clone https://github.com/yourusername/smart-traffic-management.git  
   cd smart-traffic-management  

2. Install dependencies:  
   pip install -r requirements.txt  

3. Download YOLOv8 weights:  
   wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8n.pt  

4. Run the project:  
   python main.py  

Methodology  

1. Input Acquisition  
- Captures real-time traffic footage using cameras at intersections.  

2. Vehicle Detection & Classification  
- The YOLOv8 model detects and classifies vehicles into predefined categories.  

3. Traffic Weight Calculation  
- Assigns weight to each vehicle type:  
  - Two-Wheelers: Weight = 2  
  - Three-Wheelers (Auto-rickshaws): Weight = 3  
  - Four-Wheelers (Cars): Weight = 3.5  
  - Large Vehicles (Trucks, Buses): Weight = 4  

4. Green Light Time Computation  
- Uses a custom formula to calculate the optimal green light time:  
  Traffic Weight = SUM(Vehicle Count × Weight per Vehicle)  
  Effective Green Light Time = Traffic Weight / 4  

5. Traffic Signal Control  
- The calculated green light duration is dynamically adjusted based on real-time traffic conditions.  

Future Enhancements  
- Integration with IoT sensors for real-time data collection  
- Cloud-based monitoring and analytics dashboard  
- AI-powered predictive traffic flow analysis  
- Integration with smart city frameworks  

Contributors  
- Your Name - Developer & Researcher  
- Team Members - Additional Contributors  

License  
This project is licensed under the MIT License.  

Contact  
For inquiries, feel free to reach out at:  
Email: your.email@example.com  
LinkedIn: https://www.linkedin.com/in/your-profile/  
