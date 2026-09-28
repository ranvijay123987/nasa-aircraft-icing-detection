# =============================================================================
# FLASK WEB APPLICATION - AIRCRAFT ICING DETECTION INTERFACE
# Real-time Icing Probability Prediction System
# =============================================================================

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import json

app = Flask(__name__)

# Load trained models and preprocessors
print("Loading models...")
xgb_model = joblib.load('icing_detection_xgboost.pkl')
nn_model = keras.models.load_model('icing_detection_nn.h5')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
feature_names = joblib.load('feature_names.pkl')
print("Models loaded successfully!")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def preprocess_input(input_data):
    """
    Preprocess user input to match training data format
    """
    # Create DataFrame with all features
    df = pd.DataFrame([input_data])
    
    # Add engineered features
    df['temp_humidity_interaction'] = (
        df['temperature_c'] * df['rel_humidity_pct']
    )
    df['icing_risk_index'] = (
        (df['liquid_water_content'] * df['rel_humidity_pct']) / 
        (abs(df['temperature_c']) + 1)
    )
    df['altitude_category'] = pd.cut(
        df['altitude_ft'], 
        bins=[0, 10000, 20000, 40000],
        labels=[0, 1, 2]
    ).astype(float)
    
    # Select only the features used in training
    df_selected = df[feature_names]
    
    # Apply imputation and scaling
    df_imputed = imputer.transform(df_selected)
    df_scaled = scaler.transform(df_imputed)
    
    return df_scaled

def get_risk_level(probability):
    """
    Categorize icing risk based on probability
    """
    if probability < 0.2:
        return "LOW", "success"
    elif probability < 0.5:
        return "MODERATE", "warning"
    elif probability < 0.8:
        return "HIGH", "danger"
    else:
        return "CRITICAL", "danger"

def get_recommendations(probability, input_data):
    """
    Provide actionable recommendations based on prediction
    """
    recommendations = []
    
    if probability > 0.5:
        recommendations.append("⚠️ IMMEDIATE ACTION REQUIRED")
        recommendations.append("Consider altering flight path or altitude")
        
        if input_data['altitude_ft'] < 15000:
            recommendations.append("→ Recommend climbing to higher altitude")
        elif input_data['altitude_ft'] > 25000:
            recommendations.append("→ Recommend descending to lower altitude")
        
        recommendations.append("→ Activate anti-icing systems preventively")
        recommendations.append("→ Reduce airspeed if conditions persist")
        recommendations.append("→ Notify ATC of icing conditions")
    
    elif probability > 0.3:
        recommendations.append("⚡ CAUTION: Elevated icing risk detected")
        recommendations.append("→ Monitor weather conditions closely")
        recommendations.append("→ Prepare anti-icing systems for activation")
        recommendations.append("→ Consider route deviation if possible")
    
    else:
        recommendations.append("✓ Low icing risk - conditions favorable")
        recommendations.append("→ Continue normal operations")
        recommendations.append("→ Maintain situational awareness")
    
    return recommendations

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """
    Main page with input form
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    """
    try:
        # Get input data from form
        input_data = {
            'temperature_c': float(request.form['temperature']),
            'rel_humidity_pct': float(request.form['humidity']),
            'altitude_ft': float(request.form['altitude']),
            'true_airspeed_kts': float(request.form['airspeed']),
            'liquid_water_content': float(request.form['lwc']),
            'droplet_size_micron': float(request.form['droplet_size']),
            'air_pressure_mb': float(request.form['pressure']),
            'vertical_velocity_fpm': float(request.form['vertical_velocity']),
            'angle_of_attack_deg': float(request.form['aoa']),
            'cloud_coverage_pct': float(request.form['cloud_coverage']),
            'visibility_km': float(request.form['visibility']),
            'wind_speed_kts': float(request.form['wind_speed'])
        }
        
        # Preprocess input
        processed_input = preprocess_input(input_data)
        
        # Get predictions from both models
        xgb_prob = float(xgb_model.predict_proba(processed_input)[0][1])
        nn_prob = float(nn_model.predict(processed_input)[0][0])
        
        # Ensemble prediction (average)
        ensemble_prob = (xgb_prob + nn_prob) / 2
        
        # Get risk level and recommendations
        risk_level, alert_class = get_risk_level(ensemble_prob)
        recommendations = get_recommendations(ensemble_prob, input_data)
        
        # Prepare response
        response = {
            'success': True,
            'probability': round(ensemble_prob * 100, 2),
            'xgb_probability': round(xgb_prob * 100, 2),
            'nn_probability': round(nn_prob * 100, 2),
            'risk_level': risk_level,
            'alert_class': alert_class,
            'recommendations': recommendations,
            'input_data': input_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Handle batch predictions from CSV upload
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        predictions = []
        for _, row in df.iterrows():
            processed = preprocess_input(row.to_dict())
            prob = float(xgb_model.predict_proba(processed)[0][1])
            risk_level, _ = get_risk_level(prob)
            
            predictions.append({
                'probability': round(prob * 100, 2),
                'risk_level': risk_level
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# =============================================================================
# HTML TEMPLATE (Save as templates/index.html)
# =============================================================================

INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aircraft Icing Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.0/dist/chart.min.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .header-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
        }
        .prediction-card {
            display: none;
            margin-top: 30px;
        }
        .gauge-container {
            width: 250px;
            height: 250px;
            margin: 0 auto;
        }
        .recommendation-item {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .input-group-text {
            min-width: 80px;
        }
        .alert-pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Header -->
        <div class="card header-card">
            <div class="text-center">
                <i class="bi bi-cloud-snow" style="font-size: 3rem;"></i>
                <h1 class="mt-3">Aircraft Icing Detection System</h1>
                <p class="lead">Predictive Machine Learning for Aviation Safety</p>
                <p class="mb-0"><small>Madhur Arora | J.C. Bose University of Science & Technology, YMCA</small></p>
            </div>
        </div>

        <!-- Input Form -->
        <div class="card">
            <div class="card-body p-4">
                <h4 class="card-title mb-4">
                    <i class="bi bi-clipboard-data"></i> Enter Flight Parameters
                </h4>
                <form id="predictionForm">
                    <div class="row">
                        <!-- Atmospheric Conditions -->
                        <div class="col-md-6">
                            <h5 class="text-primary mb-3">Atmospheric Conditions</h5>
                            
                            <div class="mb-3">
                                <label class="form-label">Temperature (°C)</label>
                                <div class="input-group">
                                    <span class="input-group-text">🌡️</span>
                                    <input type="number" class="form-control" name="temperature" 
                                           value="-10" step="0.1" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Relative Humidity (%)</label>
                                <div class="input-group">
                                    <span class="input-group-text">💧</span>
                                    <input type="number" class="form-control" name="humidity" 
                                           value="85" min="0" max="100" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Liquid Water Content (g/m³)</label>
                                <div class="input-group">
                                    <span class="input-group-text">💦</span>
                                    <input type="number" class="form-control" name="lwc" 
                                           value="1.2" step="0.1" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Droplet Size (μm)</label>
                                <div class="input-group">
                                    <span class="input-group-text">🔬</span>
                                    <input type="number" class="form-control" name="droplet_size" 
                                           value="25" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Air Pressure (mb)</label>
                                <div class="input-group">
                                    <span class="input-group-text">📊</span>
                                    <input type="number" class="form-control" name="pressure" 
                                           value="700" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Cloud Coverage (%)</label>
                                <div class="input-group">
                                    <span class="input-group-text">☁️</span>
                                    <input type="number" class="form-control" name="cloud_coverage" 
                                           value="80" min="0" max="100" required>
                                </div>
                            </div>
                        </div>

                        <!-- Flight Parameters -->
                        <div class="col-md-6">
                            <h5 class="text-primary mb-3">Flight Parameters</h5>
                            
                            <div class="mb-3">
                                <label class="form-label">Altitude (ft)</label>
                                <div class="input-group">
                                    <span class="input-group-text">✈️</span>
                                    <input type="number" class="form-control" name="altitude" 
                                           value="15000" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">True Airspeed (kts)</label>
                                <div class="input-group">
                                    <span class="input-group-text">🚀</span>
                                    <input type="number" class="form-control" name="airspeed" 
                                           value="250" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Vertical Velocity (fpm)</label>
                                <div class="input-group">
                                    <span class="input-group-text">📈</span>
                                    <input type="number" class="form-control" name="vertical_velocity" 
                                           value="500" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Angle of Attack (°)</label>
                                <div class="input-group">
                                    <span class="input-group-text">📐</span>
                                    <input type="number" class="form-control" name="aoa" 
                                           value="5" step="0.1" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Visibility (km)</label>
                                <div class="input-group">
                                    <span class="input-group-text">👁️</span>
                                    <input type="number" class="form-control" name="visibility" 
                                           value="5" step="0.1" required>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Wind Speed (kts)</label>
                                <div class="input-group">
                                    <span class="input-group-text">🌬️</span>
                                    <input type="number" class="form-control" name="wind_speed" 
                                           value="30" required>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="text-center mt-4">
                        <button type="submit" class="btn btn-primary btn-lg px-5">
                            <i class="bi bi-calculator"></i> Calculate Icing Probability
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <!-- Prediction Results -->
        <div class="card prediction-card" id="predictionCard">
            <div class="card-body p-4">
                <h4 class="card-title mb-4">
                    <i class="bi bi-graph-up"></i> Prediction Results
                </h4>
                
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="gauge-container">
                            <canvas id="gaugeChart"></canvas>
                        </div>
                        <h3 class="mt-3" id="probabilityText">---%</h3>
                        <div class="alert alert-pulse" id="riskAlert">
                            <h5 id="riskLevel">---</h5>
                        </div>
                    </div>

                    <div class="col-md-8">
                        <h5 class="text-primary mb-3">Model Predictions</h5>
                        <div class="mb-3">
                            <strong>Ensemble Prediction:</strong> <span id="ensembleProb">---%</span>
                        </div>
                        <div class="mb-3">
                            <strong>XGBoost Model:</strong> <span id="xgbProb">---%</span>
                        </div>
                        <div class="mb-3">
                            <strong>Neural Network:</strong> <span id="nnProb">---%</span>
                        </div>

                        <hr>

                        <h5 class="text-primary mb-3">Recommendations</h5>
                        <div id="recommendationsContainer"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let gaugeChart = null;

        $('#predictionForm').on('submit', function(e) {
            e.preventDefault();
            
            $.ajax({
                url: '/predict',
                method: 'POST',
                data: $(this).serialize(),
                success: function(response) {
                    displayResults(response);
                },
                error: function(xhr) {
                    alert('Error: ' + xhr.responseJSON.error);
                }
            });
        });

        function displayResults(data) {
            $('#probabilityText').text(data.probability + '%');
            $('#ensembleProb').text(data.probability + '%');
            $('#xgbProb').text(data.xgb_probability + '%');
            $('#nnProb').text(data.nn_probability + '%');
            $('#riskLevel').text(data.risk_level + ' RISK');
            
            const alertDiv = $('#riskAlert');
            alertDiv.removeClass('alert-success alert-warning alert-danger');
            alertDiv.addClass('alert-' + data.alert_class);
            
            updateGauge(data.probability);
            
            const recContainer = $('#recommendationsContainer');
            recContainer.empty();
            data.recommendations.forEach(rec => {
                recContainer.append(
                    '<div class="recommendation-item">' + rec + '</div>'
                );
            });
            
            $('#predictionCard').slideDown();
            $('html, body').animate({
                scrollTop: $('#predictionCard').offset().top - 20
            }, 500);
        }

        function updateGauge(probability) {
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            
            if (gaugeChart) {
                gaugeChart.destroy();
            }
            
            gaugeChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [probability, 100 - probability],
                        backgroundColor: [
                            probability < 30 ? '#28a745' :
                            probability < 60 ? '#ffc107' : '#dc3545',
                            '#e0e0e0'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    circumference: 180,
                    rotation: 270,
                    cutout: '75%',
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    }
                }
            });
        }
    </script>
</body>
</html>
'''

# Save HTML template
import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(INDEX_HTML)

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AIRCRAFT ICING DETECTION WEB INTERFACE")
    print("="*60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
