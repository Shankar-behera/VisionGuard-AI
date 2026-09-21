(() => {
  "use strict";

  const { BACKEND_URL, API_KEY } = window.APP_CONFIG;

  const video = document.getElementById("webcam");
  const canvas = document.getElementById("overlayCanvas");
  const ctx = canvas.getContext("2d");
  const captureBtn = document.getElementById("captureBtn");
  const loader = document.getElementById("loader");
  const detectionsList = document.getElementById("detectionsList");
  const modelTag = document.getElementById("modelTag");
  const latencyTag = document.getElementById("latencyTag");
  const statusPill = document.getElementById("statusPill");
  const errorBanner = document.getElementById("errorBanner");
  const cameraError = document.getElementById("cameraError");
  const cameraErrorDetail = document.getElementById("cameraErrorDetail");

  const RISK_COLORS = { High: "#ef4444", Medium: "#f97316", Low: "#10b981" };

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
  }

  function clearError() {
    errorBanner.classList.add("hidden");
    errorBanner.textContent = "";
  }

  function setStatus(text, tone) {
    const tones = {
      online: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
      offline: "bg-red-500/10 text-red-500 border-red-500/20",
      checking: "bg-slate-800 text-slate-400 border-slate-700",
    };
    statusPill.className = `flex items-center gap-2 text-[11px] px-2.5 py-1 rounded-full font-mono tracking-wide border ${tones[tone] || tones.checking}`;
    
    // Add dot indicator
    const dotColor = tone === 'online' ? 'bg-emerald-500' : (tone === 'offline' ? 'bg-red-500' : 'bg-slate-400 animate-pulse');
    statusPill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full ${dotColor}"></span><span>${text}</span>`;
  }

  async function checkBackendHealth() {
    try {
      const resp = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setStatus("API Online", "online");
      captureBtn.disabled = false;
    } catch (err) {
      setStatus("API Offline", "offline");
      captureBtn.disabled = true;
      showError(`Backend unreachable at ${BACKEND_URL}. Check status and CORS configuration.`);
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraErrorDetail.textContent = "Camera access unsupported by this browser.";
      cameraError.classList.remove("hidden");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      };
    } catch (err) {
      cameraErrorDetail.textContent =
        err.name === "NotAllowedError"
          ? "Permission denied. Allow camera access and reload."
          : `Init failed: ${err.message}`;
      cameraError.classList.remove("hidden");
    }
  }

  function drawBoundingBoxes(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    detections.forEach((item) => {
      const [ymin, xmin, ymax, xmax] = item.box_2d;
      const left = (xmin / 1000) * canvas.width;
      const top = (ymin / 1000) * canvas.height;
      const width = ((xmax - xmin) / 1000) * canvas.width;
      const height = ((ymax - ymin) / 1000) * canvas.height;
      const color = RISK_COLORS[item.risk] || "#10b981";

      // Cleaner, thinner bounding boxes
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(left, top, width, height);

      // Label background (more structured)
      ctx.fillStyle = color;
      ctx.font = "500 12px Inter, sans-serif";
      const labelText = `${item.label} (${item.risk})`;
      const textWidth = ctx.measureText(labelText).width;
      
      const labelTop = top > 22 ? top - 22 : top;
      ctx.fillRect(left, labelTop, textWidth + 12, 22);
      
      // Label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, left + 6, labelTop + 15);
    });
  }

  function renderReportUI(detections, modelName, latencyMs) {
    modelTag.textContent = modelName;
    latencyTag.textContent = `${latencyMs} ms`;
    detectionsList.innerHTML = "";

    if (!detections || detections.length === 0) {
      detectionsList.innerHTML = `<p class="text-slate-500 text-xs px-2">No items detected in current frame.</p>`;
      return;
    }

    detections.forEach((item) => {
      const color = RISK_COLORS[item.risk] || "#10b981";
      const confidence = item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "—";
      
      const card = document.createElement("div");
      // Clean, professional data row design
      card.className = "flex items-start justify-between p-3 rounded-lg border border-white/5 bg-white/[0.02]";
      card.innerHTML = `
        <div class="flex flex-col gap-1">
          <span class="text-sm font-medium text-slate-200">${escapeHtml(item.label)}</span>
          <span class="text-[10px] font-mono text-slate-500">BOX [${item.box_2d.join(", ")}]</span>
          <span class="text-[10px] text-slate-500">Confidence: ${confidence}</span>
        </div>
        <div class="shrink-0 mt-0.5">
            <span class="text-[9px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full" 
                  style="color: ${color}; background-color: ${color}15; border: 1px solid ${color}30">
              ${escapeHtml(item.risk)}
            </span>
        </div>`;
      detectionsList.appendChild(card);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  async function runInspection() {
    if (!video.videoWidth) {
      showError("Camera feed not initialized.");
      return;
    }
    clearError();

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    tempCanvas.getContext("2d").drawImage(video, 0, 0);
    const base64Image = tempCanvas.toDataURL("image/png");

    loader.classList.remove("hidden");
    captureBtn.disabled = true;

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/inspect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY,
        },
        body: JSON.stringify({ image_base64: base64Image }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      drawBoundingBoxes(result.detections);
      renderReportUI(result.detections, result.model_used, result.latency_ms);
    } catch (err) {
      showError(`Inspection failed: ${err.message}`);
    } finally {
      loader.classList.add("hidden");
      captureBtn.disabled = false;
    }
  }

  captureBtn.addEventListener("click", runInspection);

  startCamera();
  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);
})();