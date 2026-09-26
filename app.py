"""
Flask Web Application & Real-Time Dashboard Server for ESP32-S3 Smart Cardiac Monitor
"""
# cd /d "C:\Users\elang\Downloads\Final Year Project\Smart_Cardiac_Monitoring_Esp32S3_Unsupervised_Ai\Smart_Cardiac_Monitoring_Esp32S3_Unsupervised_Ai\python"
# .venv\Scripts\activate
# python app.py  


import os
import time
import json
from flask import Flask, render_template, jsonify, request, Response
from serial_handler import serial_manager

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = 'cardiac_monitor_secret_key_2026'


@app.route("/")
def index():
    """Renders the primary cardiac monitoring dashboard."""
    return render_template("index.html")


@app.route("/api/ports", methods=["GET"])
def get_ports():
    """Lists all available hardware and simulated serial COM ports."""
    ports = serial_manager.list_ports()
    return jsonify({"status": "success", "ports": ports})


@app.route("/api/connect", methods=["POST"])
def connect_port():
    """Connects to the selected serial port and baud rate."""
    data = request.get_json(silent=True) or {}
    port = data.get("port", "").strip()
    baud = int(data.get("baud", 115200))

    if not port:
        return jsonify({"status": "error", "message": "No COM port specified"}), 400

    result = serial_manager.connect(port, baud)
    return jsonify(result)


@app.route("/api/disconnect", methods=["POST"])
def disconnect_port():
    """Disconnects from active serial port."""
    result = serial_manager.disconnect()
    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns current serial connection status."""
    telemetry = serial_manager.get_telemetry()
    return jsonify({
        "connected": telemetry["connected"],
        "port": telemetry["port"],
        "baud": telemetry["baud"],
        "simulation": telemetry["simulation"],
        "is_recording": serial_manager._is_recording,
    })


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    """Returns the latest telemetry snapshot (REST fallback)."""
    return jsonify(serial_manager.get_telemetry())


@app.route("/api/stream", methods=["GET"])
def sse_telemetry_stream():
    """
    High-performance Server-Sent Events (SSE) endpoint.
    Streams 20 FPS live telemetry and PCG waveform data to the web dashboard.
    """
    def event_generator():
        while True:
            data = serial_manager.get_telemetry()
            payload = json.dumps(data)
            yield f"data: {payload}\n\n"
            time.sleep(0.05)  # 20 FPS

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/api/command", methods=["POST"])
def send_command():
    """Sends a control command to the ESP32-S3 device."""
    data = request.get_json(silent=True) or {}
    cmd = data.get("command", "").strip()
    if not cmd:
        return jsonify({"status": "error", "message": "Empty command"}), 400
    result = serial_manager.send_command(cmd)
    return jsonify(result)


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns recent serial debug logs."""
    logs = serial_manager.get_logs(limit=100)
    return jsonify({"status": "success", "logs": logs})


@app.route("/api/record/start", methods=["POST"])
def start_recording():
    """Starts session recording for CSV export."""
    return jsonify(serial_manager.start_recording())


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    """Stops session recording."""
    return jsonify(serial_manager.stop_recording())


@app.route("/api/simulation/controls", methods=["GET", "POST"])
def sim_controls_endpoint():
    """Endpoint for retrieving and updating interactive website beat control settings."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        updated = serial_manager.update_sim_controls(data)
        return jsonify({"status": "success", "controls": updated})
    else:
        return jsonify({"status": "success", "controls": serial_manager.get_sim_controls()})


@app.route("/api/train", methods=["POST"])
def trigger_training():
    """Triggers training of the Unsupervised AI Conv-AE Model via Python."""
    try:
        from train_autoencoder import train_pytorch
        success = train_pytorch()
        if success:
            meta_path = os.path.join(os.path.dirname(__file__), "models", "evt_metadata.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            return jsonify({"status": "success", "message": "Model trained successfully!", "metadata": meta})
        else:
            return jsonify({"status": "error", "message": "Training failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def get_model_info():
    """Returns information about the currently trained AI model and EVT threshold."""
    meta_path = os.path.join(os.path.dirname(__file__), "models", "evt_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return jsonify({"status": "success", "model_loaded": True, "metadata": meta})
    return jsonify({"status": "success", "model_loaded": False, "message": "No model trained yet."})


if __name__ == "__main__":
    # Automatically start live AI model simulation mode on launch
    print("=================================================================")
    print(" Smart Cardiac Monitor - Flask Web Application & AI Serial Server")
    print(" Initiating Live Conv-AE AI Model Simulation...")
    serial_manager.connect("SIMULATOR", 115200)
    print(" Starting server on http://localhost:5000 ...")
    print("=================================================================")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

