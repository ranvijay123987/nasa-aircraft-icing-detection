# NASA Aircraft In-Flight Atmospheric Icing Detection & Prediction System

An aeronautical machine learning and real-time flight safety classification platform leveraging NASA Icing Research Tunnel (IRT) sensor telemetry to predict atmospheric icing hazards, ice accretion severity, and alert flight operations in real time.

---

## 🌟 Overview & Significance

Atmospheric icing is one of aviation's most hazardous phenomena, causing loss of aerodynamic lift, increased drag, Pitot-static tube blockage, and sensor failures. This project models multi-dimensional meteorological and aerodynamic telemetry to predict icing onset and severity before critical structural accumulation occurs.

---

## 🔬 Scientific Data & Features

Trained on sensor measurements from the **NASA Icing Research Tunnel (IRT)** dataset:
- **Temperature (\	emperature_c\)**: Ambient and total air temperature (°C).
- **Liquid Water Content (\liquid_water_content\)**: Supercooled liquid water density (g/m³).
- **Median Volume Diameter (\droplet_diameter_mvd\)**: Cloud droplet size distribution (μm).
- **True Airspeed (\	rue_airspeed_ms\)**: Aerodynamic relative velocity (m/s).
- **Static Pressure & Humidity**: Altitude-dependent thermodynamic states.
- **Engineered Telemetry**:
  - \	emp_humidity_interaction\
  - \icing_risk_index\: Non-linear composite risk metric combining supercooling, water density, and droplet dynamics.

---

## 🤖 Model Architectures

1. **Random Forest Classifier**: Non-parametric ensemble capable of handling multi-collinear meteorological inputs and non-linear icing boundaries.
2. **XGBoost Classifier**: Gradient-boosted decision trees for low-latency scoring and precision calibration.
3. **Deep Neural Network**: Multi-layer perceptron mapping continuous sensor inputs to hazard probabilities.

---

## 🌐 Real-Time Flight Deck Dashboard (Flask Web App)

The application provides an interactive web cockpit interface:
- **Telemetry Input Panel**: Enter in-flight sensor readings (Temperature, Humidity, LWC, Airspeed, Droplet MVD).
- **Hazard Probability Gauge**: Dynamic probability breakdown with color-coded safety levels (Safe, Trace, Light, Moderate, Severe).
- **Physical Parameter Insights**: Instant explanation of factors driving the risk score.

---

## 🚀 Getting Started

### 1. Clone the Repository
\\\ash
git clone https://github.com/ranvijay123987/nasa-aircraft-icing-detection.git
cd nasa-aircraft-icing-detection
\\\

### 2. Install Dependencies
\\\ash
python -m venv venv
venv\Scripts\activate   # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
\\\

### 3. Train the Model
\\\ash
python aircraft_icing_training.py
\\\

### 4. Run the Real-Time Dashboard
\\\ash
python flask_app.py
\\\
Navigate to [http://localhost:5000](http://localhost:5000) to view the telemetry dashboard.

---

## 📄 License
This project is open-source and available under the MIT License.
