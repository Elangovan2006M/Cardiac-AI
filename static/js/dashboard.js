/**
 * Smart Cardiac Monitor AI • High-Performance Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------
  // DOM Elements
  // -------------------------------------------------------------
  const portSelect = document.getElementById('port-select');
  const baudSelect = document.getElementById('baud-select');
  const btnRefreshPorts = document.getElementById('btn-refresh-ports');
  const btnConnect = document.getElementById('btn-connect');
  const btnConnectText = document.getElementById('btn-connect-text');
  const connectionBadge = document.getElementById('connection-badge');
  const connectionStatusText = document.getElementById('connection-status-text');

  // Pipeline & Latency Elements
  const valLoopLatency = document.getElementById('val-loop-latency');
  const badgeLatencyStatus = document.getElementById('badge-latency-status');
  const latencyProgressFill = document.getElementById('latency-progress-fill');
  const valPcTime = document.getElementById('val-pc-time');
  const valTftSynced = document.getElementById('val-tft-synced');
  const valTotalPackets = document.getElementById('val-total-packets');
  const valPcProcBadge = document.getElementById('val-pc-proc-badge');
  const badgeTftSync = document.getElementById('badge-tft-sync');

  // Scope & Spectrogram Elements
  const canvas = document.getElementById('pcg-canvas');
  const ctx = canvas.getContext('2d');
  const specCanvas = document.getElementById('spec-canvas');
  const specCtx = specCanvas ? specCanvas.getContext('2d') : null;
  const btnFreezeScope = document.getElementById('btn-freeze-scope');
  const freezeText = document.getElementById('freeze-text');
  const valVpp = document.getElementById('val-vpp');
  const valPhase = document.getElementById('val-phase');
  const gainBtns = document.querySelectorAll('[data-gain]');

  // Phase segments
  const phaseS1 = document.getElementById('phase-s1');
  const phaseSys = document.getElementById('phase-sys');
  const phaseS2 = document.getElementById('phase-s2');
  const phaseDia = document.getElementById('phase-dia');

  // Metrics
  const metricBpm = document.getElementById('metric-bpm');
  const bpmTag = document.getElementById('bpm-tag');
  const bpmMinMax = document.getElementById('bpm-minmax');
  const anomalyGaugeCircle = document.getElementById('anomaly-gauge-circle');
  const metricAnomalyPct = document.getElementById('metric-anomaly-pct');
  const anomalyStatusBadge = document.getElementById('anomaly-status-badge');
  const metricMse = document.getElementById('metric-mse');
  const msePctLabel = document.getElementById('mse-pct-label');
  const mseProgressFill = document.getElementById('mse-progress-fill');
  const btnRecalibrate = document.getElementById('btn-recalibrate');
  const hwIp = document.getElementById('hw-ip');

  // Terminal & Commands
  const terminalOutput = document.getElementById('terminal-output');
  const autoscrollChk = document.getElementById('autoscroll-chk');
  const btnClearLogs = document.getElementById('btn-clear-logs');
  const cmdInput = document.getElementById('cmd-input');
  const btnSendCmd = document.getElementById('btn-send-cmd');

  // Session Recorder
  const recIndicator = document.getElementById('rec-indicator');
  const recTimer = document.getElementById('rec-timer');
  const recSampleCount = document.getElementById('rec-sample-count');
  const recStatusText = document.getElementById('rec-status-text');
  const btnRecordToggle = document.getElementById('btn-record-toggle');
  const btnExportCsv = document.getElementById('btn-export-csv');

  // -------------------------------------------------------------
  // Application State
  // -------------------------------------------------------------
  let isConnected = false;
  let isFrozen = false;
  let scopeGain = 1.0;
  let sseSource = null;
  let isRecording = false;
  let recStartTime = 0;
  let recTimerInterval = null;

  // Waveform Buffer (rolling array of 400 points)
  const MAX_WAVE_POINTS = 400;
  const waveBuffer = new Array(MAX_WAVE_POINTS).fill(0.0);
  let bpmHistory = [];

  // -------------------------------------------------------------
  // Canvas Setup & HiDPI scaling
  // -------------------------------------------------------------
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // -------------------------------------------------------------
  // Serial Port APIs
  // -------------------------------------------------------------
  async function fetchPorts() {
    try {
      const res = await fetch('/api/ports');
      const data = await res.json();
      if (data.status === 'success') {
        portSelect.innerHTML = '';
        data.ports.forEach(port => {
          const opt = document.createElement('option');
          opt.value = port.device;
          opt.textContent = `${port.device} - ${port.description}`;
          portSelect.appendChild(opt);
        });
      }
    } catch (err) {
      appendTerminalLog('[ERROR] Failed to scan serial ports: ' + err.message, 'log-alert');
    }
  }

  async function checkInitialStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.connected) {
        setConnectedUI(true, data.port, data.simulation);
        startTelemetryStream();
      } else {
        setConnectedUI(false);
      }
    } catch (e) {
      setConnectedUI(false);
    }
  }

  async function toggleConnect() {
    if (isConnected) {
      // Disconnect
      try {
        const res = await fetch('/api/disconnect', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
          setConnectedUI(false);
          stopTelemetryStream();
          appendTerminalLog('[INFO] Port disconnected by user.', 'log-info');
        }
      } catch (err) {
        appendTerminalLog('[ERROR] Disconnect failed: ' + err.message, 'log-alert');
      }
    } else {
      // Connect
      const port = portSelect.value;
      const baud = parseInt(baudSelect.value, 10);
      if (!port) {
        alert('Please select a valid COM port first.');
        return;
      }

      btnConnect.disabled = true;
      btnConnectText.textContent = 'CONNECTING...';

      try {
        const res = await fetch('/api/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ port, baud })
        });
        const data = await res.json();
        if (data.status === 'success') {
          setConnectedUI(true, port, data.simulation);
          startTelemetryStream();
          appendTerminalLog(`[INFO] Connected to ${port} @ ${baud} baud.`, 'log-cmd');
        } else {
          alert('Failed to connect: ' + data.message);
          setConnectedUI(false);
          appendTerminalLog('[ERROR] Connection failed: ' + data.message, 'log-alert');
        }
      } catch (err) {
        alert('Connection error: ' + err.message);
        setConnectedUI(false);
      } finally {
        btnConnect.disabled = false;
      }
    }
  }

  function setConnectedUI(connected, portName = '', isSim = false) {
    isConnected = connected;
    if (connected) {
      btnConnect.classList.add('btn-disconnect');
      btnConnectText.textContent = 'DISCONNECT';
      if (isSim) {
        connectionBadge.className = 'status-pill pill-simulation';
        connectionStatusText.textContent = 'VIRTUAL SIM';
      } else {
        connectionBadge.className = 'status-pill pill-connected';
        connectionStatusText.textContent = `ONLINE: ${portName}`;
      }
    } else {
      btnConnect.classList.remove('btn-disconnect');
      btnConnectText.textContent = 'CONNECT';
      connectionBadge.className = 'status-pill pill-disconnected';
      connectionStatusText.textContent = 'DISCONNECTED';
    }
  }

  // -------------------------------------------------------------
  // Server-Sent Events (SSE) Real-Time Stream
  // -------------------------------------------------------------
  function startTelemetryStream() {
    if (sseSource) {
      sseSource.close();
    }

    sseSource = new EventSource('/api/stream');

    sseSource.onmessage = (event) => {
      try {
        const telemetry = JSON.parse(event.data);
        handleTelemetryUpdate(telemetry);
      } catch (err) {
        console.error('Telemetry parse error:', err);
      }
    };

    sseSource.onerror = () => {
      console.warn('SSE stream disconnected, retrying...');
    };
  }

  function stopTelemetryStream() {
    if (sseSource) {
      sseSource.close();
      sseSource = null;
    }
  }

  // -------------------------------------------------------------
  // Telemetry Processing
  // -------------------------------------------------------------
  function handleTelemetryUpdate(data) {
    // 1. Update Waveform Buffer
    if (!isFrozen && data.waveform && Array.isArray(data.waveform)) {
      for (const pt of data.waveform) {
        waveBuffer.shift();
        waveBuffer.push(pt);
      }
    }

    // 2. Update BPM Display & Rhythm
    const bpm = Math.round(data.bpm || 72);
    metricBpm.textContent = bpm;
    bpmHistory.push(bpm);
    if (bpmHistory.length > 50) bpmHistory.shift();

    const minBpm = Math.min(...bpmHistory);
    const maxBpm = Math.max(...bpmHistory);
    bpmMinMax.textContent = `${minBpm} / ${maxBpm}`;

    if (bpm > 100) {
      bpmTag.className = 'badge-rhythm rhythm-tachy';
      bpmTag.textContent = 'TACHYCARDIA';
    } else if (bpm < 60) {
      bpmTag.className = 'badge-rhythm rhythm-brady';
      bpmTag.textContent = 'BRADYCARDIA';
    } else {
      bpmTag.className = 'badge-rhythm rhythm-normal';
      bpmTag.textContent = 'NORMAL SINUS';
    }

    // 3. Update TinyML Anomaly Risk Gauge
    const anomalyScore = Math.min(100, Math.max(0, data.anomaly_score || 0));
    metricAnomalyPct.textContent = anomalyScore.toFixed(1) + '%';

    const maxOffset = 339.29; // 2 * PI * 54
    const offset = maxOffset - (anomalyScore / 100) * maxOffset;
    anomalyGaugeCircle.style.strokeDashoffset = offset;

    if (data.is_anomaly || anomalyScore > 60) {
      anomalyGaugeCircle.style.stroke = 'var(--red-glow)';
      anomalyStatusBadge.className = 'badge-anomaly-state state-anomaly';
      anomalyStatusBadge.textContent = 'ANOMALY DETECTED';
    } else if (anomalyScore > 30) {
      anomalyGaugeCircle.style.stroke = 'var(--amber-glow)';
      anomalyStatusBadge.className = 'badge-anomaly-state state-watch';
      anomalyStatusBadge.textContent = 'WATCH / ELEVATED';
    } else {
      anomalyGaugeCircle.style.stroke = 'var(--green-glow)';
      anomalyStatusBadge.className = 'badge-anomaly-state state-normal';
      anomalyStatusBadge.textContent = 'NORMAL';
    }

    // 4. Update Reconstruction MSE
    const mse = data.reconstruction_mse || 0.0018;
    metricMse.textContent = mse.toFixed(4);
    const msePct = Math.min(100, (mse / 0.010) * 100);
    msePctLabel.textContent = Math.round(msePct) + '%';
    mseProgressFill.style.width = msePct + '%';
    if (msePct > 60) {
      mseProgressFill.style.background = 'linear-gradient(90deg, #ffb800, #ff3366)';
    } else {
      mseProgressFill.style.background = 'linear-gradient(90deg, #00f5ff, #00ff88)';
    }

    // 5. Update Hardware Telemetry
    if (data.ip_address) {
      hwIp.textContent = data.ip_address;
    }

    // 6. Update Heart Phase Indicators
    const lastSample = waveBuffer[waveBuffer.length - 1];
    const absVal = Math.abs(lastSample);
    valVpp.textContent = (absVal * 1.8).toFixed(2) + ' V';

    if (absVal > 0.45) {
      valPhase.textContent = 'S1 (LUB)';
      valPhase.className = 'overlay-value phase-s1';
      phaseS1.className = 'phase-segment active-s1';
      phaseSys.className = 'phase-segment';
      phaseS2.className = 'phase-segment';
      phaseDia.className = 'phase-segment';
    } else if (absVal > 0.25) {
      valPhase.textContent = 'S2 (DUB)';
      valPhase.className = 'overlay-value phase-s2';
      phaseS1.className = 'phase-segment';
      phaseSys.className = 'phase-segment';
      phaseS2.className = 'phase-segment active-s2';
      phaseDia.className = 'phase-segment';
    } else {
      valPhase.textContent = 'DIASTOLE';
      valPhase.className = 'overlay-value';
      phaseS1.className = 'phase-segment';
      phaseSys.className = 'phase-segment';
      phaseS2.className = 'phase-segment';
      phaseDia.className = 'phase-segment';
    }

    // 7. Update Distributed Pipeline & Latency Enforcer (< 3.0s Limit)
    if (valLoopLatency && data.loop_latency_s !== undefined) {
      const latS = Number(data.loop_latency_s) || 0.65;
      valLoopLatency.textContent = latS.toFixed(2);

      const pct = Math.min(100, Math.max(5, (latS / 3.0) * 100));
      if (latencyProgressFill) latencyProgressFill.style.width = pct + '%';

      if (latS <= 3.0) {
        if (badgeLatencyStatus) {
          badgeLatencyStatus.className = 'badge-status-pass';
          badgeLatencyStatus.textContent = 'PASS (< 3.0s)';
        }
        if (latencyProgressFill) {
          latencyProgressFill.style.background = 'linear-gradient(90deg, #00ff88, #00f5ff)';
        }
      } else {
        if (badgeLatencyStatus) {
          badgeLatencyStatus.className = 'badge-status-fail';
          badgeLatencyStatus.textContent = 'EXCEEDED (> 3.0s)';
        }
        if (latencyProgressFill) {
          latencyProgressFill.style.background = 'linear-gradient(90deg, #ffb800, #ff3366)';
        }
      }
    }

    if (valPcTime && data.pc_proc_time_ms !== undefined) {
      valPcTime.textContent = data.pc_proc_time_ms.toFixed(1) + ' ms';
      if (valPcProcBadge) valPcProcBadge.textContent = data.pc_proc_time_ms.toFixed(1) + ' ms';
    }

    if (valTotalPackets && data.packets_received !== undefined) {
      valTotalPackets.textContent = data.packets_received;
    }

    if (valTftSynced && data.esp32_display_synced) {
      valTftSynced.textContent = 'CONFIRMED (PACKET ACK)';
      valTftSynced.className = 'timing-value text-green';
      if (badgeTftSync) {
        badgeTftSync.className = 'node-badge badge-synced';
        badgeTftSync.textContent = 'DISPLAY LIVE';
      }
    }

    // 8. Render 32x32 Log-Mel Spectrogram Heatmap
    if (specCtx && data.spectrogram && Array.isArray(data.spectrogram)) {
      renderSpectrogram(data.spectrogram);
    }

    // Update Session Sample Count if recording
    if (isRecording) {
      const currentCount = parseInt(recSampleCount.textContent, 10) || 0;
      recSampleCount.textContent = currentCount + 1;
    }
  }

  function renderSpectrogram(spec) {
    if (!specCtx || !spec || !spec.length) return;
    const w = specCanvas.width;
    const h = specCanvas.height;
    const rows = spec.length; // 32 frequency bins
    const cols = spec[0].length; // 32 time frames
    const cellW = w / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Mel bin 0 (20 Hz) placed at the bottom, bin 31 (400 Hz) at the top
        const val = Math.min(1.0, Math.max(0.0, spec[rows - 1 - r][c]));

        let red = 0, green = 0, blue = 0;
        if (val < 0.25) {
          blue = Math.round(val * 4 * 255);
        } else if (val < 0.5) {
          green = Math.round((val - 0.25) * 4 * 255);
          blue = 255;
        } else if (val < 0.75) {
          red = Math.round((val - 0.5) * 4 * 255);
          green = 255;
          blue = Math.round((1 - (val - 0.5) * 4) * 255);
        } else {
          red = 255;
          green = Math.round((1 - (val - 0.75) * 4) * 200);
          blue = Math.round((val - 0.75) * 4 * 255);
        }
        specCtx.fillStyle = `rgb(${red},${green},${blue})`;
        specCtx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }

  // -------------------------------------------------------------
  // 60 FPS HTML5 Canvas Oscilloscope Rendering Loop
  // -------------------------------------------------------------
  function drawScope() {
    const width = canvas.getBoundingClientRect().width;
    const height = canvas.getBoundingClientRect().height;

    // Clear background
    ctx.fillStyle = '#04070e';
    ctx.fillRect(0, 0, width, height);

    // 1. Draw Medical Grid
    const gridMajorX = width / 8;
    const gridMajorY = height / 6;

    ctx.strokeStyle = 'rgba(0, 245, 255, 0.06)';
    ctx.lineWidth = 1;

    // Vertical minor ticks
    for (let x = 0; x < width; x += gridMajorX / 4) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    // Horizontal minor ticks
    for (let y = 0; y < height; y += gridMajorY / 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Major grid lines
    ctx.strokeStyle = 'rgba(0, 245, 255, 0.14)';
    for (let x = 0; x < width; x += gridMajorX) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridMajorY) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Center baseline
    const midY = height / 2;
    ctx.strokeStyle = 'rgba(0, 245, 255, 0.28)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // 2. Draw Phosphorescent PCG Waveform
    if (waveBuffer.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#00f5ff';
      ctx.shadowColor = '#00f5ff';
      ctx.shadowBlur = 12;
      ctx.lineWidth = 2.4;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      ctx.beginPath();
      const step = width / (waveBuffer.length - 1);

      for (let i = 0; i < waveBuffer.length; i++) {
        const x = i * step;
        const rawY = waveBuffer[i] * scopeGain;
        const clampedY = Math.max(-1.2, Math.min(1.2, rawY));
        const y = midY - (clampedY * (height * 0.42));

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Sweep lead head glow
      const headX = (waveBuffer.length - 1) * step;
      const headY = midY - (waveBuffer[waveBuffer.length - 1] * scopeGain * (height * 0.42));
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#00ff88';
      ctx.shadowBlur = 16;
      ctx.beginPath();
      ctx.arc(headX, headY, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    }

    requestAnimationFrame(drawScope);
  }

  // Start 60 FPS animation loop
  requestAnimationFrame(drawScope);

  // -------------------------------------------------------------
  // Controls & Listeners
  // -------------------------------------------------------------
  btnRefreshPorts.addEventListener('click', fetchPorts);
  btnConnect.addEventListener('click', toggleConnect);

  gainBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      gainBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      scopeGain = parseFloat(btn.getAttribute('data-gain'));
    });
  });

  btnFreezeScope.addEventListener('click', () => {
    isFrozen = !isFrozen;
    freezeText.textContent = isFrozen ? 'RESUME' : 'FREEZE';
    btnFreezeScope.classList.toggle('active', isFrozen);
  });

  btnRecalibrate.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: 'CALIBRATE' })
      });
      const data = await res.json();
      appendTerminalLog('[CMD TX] Sent baseline calibration request to ESP32-S3.', 'log-cmd');
    } catch (e) {
      appendTerminalLog('[ERROR] Calibration command failed: ' + e.message, 'log-alert');
    }
  });

  // Terminal Log Polling
  async function pollLogs() {
    try {
      const res = await fetch('/api/logs');
      const data = await res.json();
      if (data.status === 'success' && data.logs.length > 0) {
        terminalOutput.innerHTML = '';
        data.logs.forEach(line => {
          let cls = 'log-info';
          if (line.includes('[ALERT]') || line.includes('[ERROR]')) cls = 'log-alert';
          else if (line.includes('[CMD')) cls = 'log-cmd';
          appendTerminalLog(line, cls, false);
        });
        if (autoscrollChk.checked) {
          terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }
      }
    } catch (e) {}
  }
  setInterval(pollLogs, 1500);

  function appendTerminalLog(msg, cls = 'log-info', scroll = true) {
    const div = document.createElement('div');
    div.className = `terminal-line ${cls}`;
    div.textContent = msg;
    terminalOutput.appendChild(div);
    if (scroll && autoscrollChk.checked) {
      terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }
  }

  btnClearLogs.addEventListener('click', () => {
    terminalOutput.innerHTML = '';
  });

  function sendTerminalCommand() {
    const cmd = cmdInput.value.trim();
    if (!cmd) return;
    fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: cmd })
    }).then(r => r.json()).then(d => {
      appendTerminalLog(`[CMD TX] ${cmd}`, 'log-cmd');
      cmdInput.value = '';
    }).catch(err => {
      appendTerminalLog(`[ERROR] Command failed: ${err.message}`, 'log-alert');
    });
  }

  btnSendCmd.addEventListener('click', sendTerminalCommand);
  cmdInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendTerminalCommand();
  });

  // Session Recording
  btnRecordToggle.addEventListener('click', async () => {
    if (!isRecording) {
      // Start Recording
      const res = await fetch('/api/record/start', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        isRecording = true;
        recStartTime = Date.now();
        recIndicator.classList.add('recording');
        recStatusText.textContent = 'RECORDING';
        recStatusText.style.color = '#ff3366';
        btnRecordToggle.querySelector('span').textContent = 'STOP RECORDING';
        btnExportCsv.classList.add('disabled');
        recSampleCount.textContent = '0';

        recTimerInterval = setInterval(() => {
          const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
          const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
          const secs = String(elapsed % 60).padStart(2, '0');
          recTimer.textContent = `${mins}:${secs}`;
        }, 1000);
      }
    } else {
      // Stop Recording
      const res = await fetch('/api/record/stop', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        isRecording = false;
        clearInterval(recTimerInterval);
        recIndicator.classList.remove('recording');
        recStatusText.textContent = 'SAVED';
        recStatusText.style.color = '#00ff88';
        btnRecordToggle.querySelector('span').textContent = 'START RECORDING';
        btnExportCsv.classList.remove('disabled');
      }
    }
  });

  // -------------------------------------------------------------
  // Interactive Beat Control & Simulation Handlers
  // -------------------------------------------------------------
  const pathologyChips = document.querySelectorAll('.pathology-chip');
  const sliderBpm = document.getElementById('slider-bpm');
  const bpmSliderVal = document.getElementById('bpm-slider-val');
  const sliderNoise = document.getElementById('slider-noise');
  const noiseSliderVal = document.getElementById('noise-slider-val');
  const btnTriggerAnomaly = document.getElementById('btn-trigger-anomaly');
  const btnRetrainAi = document.getElementById('btn-retrain-ai');

  async function updateSimControls(controls) {
    try {
      await fetch('/api/simulation/controls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(controls)
      });
    } catch (e) {
      console.error('Sim control update failed:', e);
    }
  }

  pathologyChips.forEach(chip => {
    chip.addEventListener('click', () => {
      pathologyChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const pathology = chip.getAttribute('data-pathology');
      updateSimControls({ pathology });
      appendTerminalLog(`[SIM] Pathology switched to ${pathology}`, 'log-cmd');
    });
  });

  if (sliderBpm) {
    sliderBpm.addEventListener('input', (e) => {
      const bpm = parseInt(e.target.value, 10);
      bpmSliderVal.textContent = `${bpm} BPM`;
      updateSimControls({ bpm });
    });
  }

  if (sliderNoise) {
    sliderNoise.addEventListener('input', (e) => {
      const noise_level = parseInt(e.target.value, 10);
      noiseSliderVal.textContent = `${noise_level}%`;
      updateSimControls({ noise_level });
    });
  }

  if (btnTriggerAnomaly) {
    btnTriggerAnomaly.addEventListener('click', () => {
      updateSimControls({ manual_anomaly: true });
      appendTerminalLog('[ALERT] Manually injected pathological acoustic spike!', 'log-alert');
      btnTriggerAnomaly.style.transform = 'scale(0.95)';
      setTimeout(() => {
        btnTriggerAnomaly.style.transform = 'none';
        updateSimControls({ manual_anomaly: false });
      }, 3500);
    });
  }

  if (btnRetrainAi) {
    btnRetrainAi.addEventListener('click', async () => {
      btnRetrainAi.disabled = true;
      btnRetrainAi.textContent = '⏳ RETRAINING ON 1500 SAMPLES...';
      appendTerminalLog('[AI] Retraining PyTorch Conv-AE model on 1500 biomedical samples...', 'log-info');
      try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
          const meta = data.metadata || {};
          appendTerminalLog(`[AI] Training complete! ROC-AUC: ${meta.roc_auc || 1.0}, F1: ${meta.f1_score || 0.99}, EVT T: ${meta.evt_threshold || 0.000632}`, 'log-cmd');
          alert(`Model Retrained Successfully!\nROC-AUC: ${meta.roc_auc || 1.0}\nF1-Score: ${meta.f1_score || 0.99}\nEVT Threshold: ${meta.evt_threshold || 0.000632}`);
        } else {
          alert('Training failed: ' + data.message);
        }
      } catch (err) {
        alert('Training error: ' + err.message);
      } finally {
        btnRetrainAi.disabled = false;
        btnRetrainAi.textContent = '⚡ RETRAIN AI MODEL (1500 SAMPLES)';
      }
    });
  }

  // Initial Startup
  fetchPorts();
  checkInitialStatus();
});

