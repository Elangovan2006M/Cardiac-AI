# Cardiac-AI

Local demonstration dashboard for ESP32-S3 PCG streaming and a synthetic-trained Conv-AE model.

## Run

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

## Safety notice

This project is a software demonstration only. It is not a medical device, is not clinically validated, and must not be used for diagnosis or patient-care decisions.

## Serial protocol

Binary packets use the header `<4sHIH` followed by signed 16-bit PCM samples and a firmware-defined checksum. The checksum algorithm must be shared explicitly by firmware and host; this repository will reject packets with an invalid checksum only when firmware emits the documented CRC mode.

## Tests

```bash
python -m unittest discover -s tests -v
```
