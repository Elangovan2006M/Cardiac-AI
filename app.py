"""
Flask Web Application & Real-Time Dashboard Server for ESP32-S3 Smart Cardiac Monitor
"""
import os
import time
import json
import threading
from flask import Flask, render_template, jsonify, request, Response
from serial_handler import serial_manager

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = os.environ.get('CARDIAC_AI_SECRET_KEY', os.urandom(32))

_training_lock = threading.Lock()
_supported_baud_rates = {9600, 230400, 460800, 921600, 115200}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ports", methods=["GET"])
def get_ports():
    return jsonify({"status": "success", "ports": serial_manager.list_ports()})


@app.route("/api/connect", methods=["POST"])
def connect_port():
    data = request.get_json(silent=True) or {}
    port = str(data.get("port", "")).strip()
    if not port:
        return jsonify({"status": "error", "message": "No COM port specified"}), 400
    try:
        baud = int(data.get("baud", 115200))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid baud rate"}), 400
    if baud not in _supported_baud_rates:
        return jsonify({"status": "error", "message": "Unsupported baud rate"}), 400
    return jsonify(serial_manager.connect(port, baud))


@app.route("/api/disconnect", methods=["POST"])
def disconnect_port():
    return jsonify(serial_manager.disconnect())


@app.route("/api/status", methods=["GET"])
def get_status():
    telemetry = serial_manager.get_telemetry()
    return jsonify({
        "connected": telemetry["connected"],
        "port": telemetry["port"],
        "baud": telemetry["baud"],
        "simulation": telemetry["simulation"],
        "is_recording": serial_manager.is_recording(),
    })


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    return jsonify(serial_manager.get_telemetry())


@app.route("/api/stream", methods=["GET"])
def sse_telemetry_stream():
    def event_generator():
        while True:
            yield f"data: {json.dumps(serial_manager.get_telemetry())}\n\n"
            time.sleep(0.05)

    return Response(event_generator(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/api/command", methods=["POST"])
def send_command():
    data = request.get_json(silent=True) or {}
    cmd = str(data.get("command", "")).strip()
    if not cmd:
        return jsonify({"status": "error", "message": "Empty command"}), 400
    if len(cmd) > 256:
        return jsonify({"status": "error", "message": "Command too long"}), 400
    return jsonify(serial_manager.send_command(cmd))


@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify({"status": "success", "logs": serial_manager.get_logs(limit=100)})


@app.route("/api/record/start", methods=["POST"])
def start_recording():
    return jsonify(serial_manager.start_recording())


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    return jsonify(serial_manager.stop_recording())


@app.route("/api/record/export", methods=["GET"])
def export_recording():
    csv_text = serial_manager.export_csv()
    return Response(csv_text, mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=cardiac_telemetry.csv"
    })


@app.route("/api/simulation/controls", methods=["GET", "POST"])
def sim_controls_endpoint():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        updated = serial_manager.update_sim_controls(data)
        return jsonify({"status": "success", "controls": updated})
    return jsonify({"status": "success", "controls": serial_manager.get_sim_controls()})


@app.route("/api/train", methods=["POST"])
def trigger_training():
    if not _training_lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "Training is already in progress"}), 409
    try:
        from train_autoencoder import train_pytorch
        success = train_pytorch()
        if not success:
            return jsonify({"status": "error", "message": "Training failed"}), 500
        meta_path = os.path.join(os.path.dirname(__file__), "models", "evt_metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return jsonify({"status": "success", "message": "Model trained successfully!", "metadata": meta})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        _training_lock.release()


@app.route("/api/model_info", methods=["GET"])
def get_model_info():
    meta_path = os.path.join(os.path.dirname(__file__), "models", "evt_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return jsonify({"status": "success", "model_loaded": True, "metadata": meta})
    return jsonify({"status": "success", "model_loaded": False, "message": "No model trained yet."})


if __name__ == "__main__":
    print("Starting Smart Cardiac Monitor simulation on http://127.0.0.1:5000")
    serial_manager.connect("SIMULATOR", 115200)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
