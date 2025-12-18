const USE_MOCK = true;

const API_URL = "http://localhost:8000/predict";

const fileInput = document.getElementById("fileInput");
const btnSubirImagen = document.getElementById("btnSubirImagen");
const statusMessage = document.getElementById("statusMessage");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");

const noResults = document.getElementById("noResults");
const resultsContainer = document.getElementById("resultsContainer");
const predictedLabel = document.getElementById("predictedLabel");
const predictedConfidence = document.getElementById("predictedConfidence");
const metricsBody = document.getElementById("metricsBody");

const menuToggle = document.getElementById("menuToggle");
const menuDropdown = document.getElementById("menuDropdown");
const btnVerHistorial = document.getElementById("btnVerHistorial");
const historyPanel = document.getElementById("historyPanel");
const historyContent = document.getElementById("historyContent");
const closeHistory = document.getElementById("closeHistory");

btnSubirImagen.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewContainer.classList.remove("hidden");
  };
  reader.readAsDataURL(file);

  analyzeImage(file);
});

async function analyzeImage(file) {
  statusMessage.textContent = "Analizando imagen...";
  statusMessage.classList.remove("error");

  if (USE_MOCK) {
    setTimeout(() => {
      const fakeResponse = {
        prediction: {
          label: "black_sigatoka",
          confidence: 0.87,
        },
        models: [
          {
            name: "ConvNeXt",
            metric_name: "F1-score",
            metric_value: 0.9421,
          },
          {
            name: "ViT",
            metric_name: "F1-score",
            metric_value: 0.9312,
          },
          {
            name: "YOLOv8",
            metric_name: "mAP@0.5",
            metric_value: 0.8923,
          },
        ],
      };

      renderResults(fakeResponse, file.name);
      saveToHistory(fakeResponse, file.name);
      statusMessage.textContent =
        "Análisis simulado (modo demo, sin backend).";
    }, 800);

    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Error HTTP ${response.status}`);
    }

    const data = await response.json();
    renderResults(data, file.name);
    saveToHistory(data, file.name);
    statusMessage.textContent = "Análisis completado.";
  } catch (err) {
    console.error(err);
    statusMessage.textContent =
      "Ocurrió un error al analizar la imagen. Revisa que el backend esté ejecutándose.";
    statusMessage.classList.add("error");
  }
}

function renderResults(data, filename) {
  noResults.classList.add("hidden");
  resultsContainer.classList.remove("hidden");

  const pred = data.prediction || {};
  predictedLabel.textContent = prettyLabel(pred.label || "Desconocida");

  predictedConfidence.textContent =
    pred.confidence != null
      ? (pred.confidence * 100).toFixed(2) + " %"
      : "N/A";

  metricsBody.innerHTML = "";

  const models = Array.isArray(data.models) ? data.models : [];
  models.forEach((m) => {
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.textContent = m.name || "-";

    const tdMetricName = document.createElement("td");
    tdMetricName.textContent = m.metric_name || "-";

    const tdMetricValue = document.createElement("td");
    tdMetricValue.textContent =
      m.metric_value != null ? m.metric_value.toFixed(4) : "N/A";

    tr.appendChild(tdName);
    tr.appendChild(tdMetricName);
    tr.appendChild(tdMetricValue);

    metricsBody.appendChild(tr);
  });
}

function prettyLabel(raw) {
  if (!raw) return "";
  const map = {
    healthy: "Hoja sana",
    black_sigatoka: "Sigatoka negra",
    sigatoka_negra: "Sigatoka negra",
    fusarium_r4t: "Fusarium R4T",
    moko: "Moko bacteriano",
    moko_disease: "Moko bacteriano",
  };
  const key = raw.toLowerCase();
  if (map[key]) return map[key];

  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}


function saveToHistory(data, filename) {
  const pred = data.prediction || {};
  const record = {
    timestamp: new Date().toISOString(),
    filename: filename,
    label: pred.label || "desconocida",
    confidence: pred.confidence ?? null,
  };

  const existing =
    JSON.parse(localStorage.getItem("banana_disease_history") || "[]") || [];
  existing.unshift(record);
  const trimmed = existing.slice(0, 20);
  localStorage.setItem("banana_disease_history", JSON.stringify(trimmed));
}

function loadHistory() {
  const existing =
    JSON.parse(localStorage.getItem("banana_disease_history") || "[]") || [];

  if (!existing.length) {
    historyContent.innerHTML =
      "<p>No existe historial almacenado en este navegador.</p>";
    return;
  }

  const list = document.createElement("div");
  existing.forEach((r) => {
    const item = document.createElement("div");
    item.className = "history-list-item";

    const date = new Date(r.timestamp);
    const fecha = date.toLocaleString("es-EC", {
      dateStyle: "short",
      timeStyle: "short",
    });

    const conf =
      r.confidence != null
        ? (r.confidence * 100).toFixed(1) + " %"
        : "N/A";

    item.innerHTML = `
      <div><strong>${fecha}</strong></div>
      <div>Archivo: <code>${r.filename}</code></div>
      <div>Etiqueta: <span class="label">${prettyLabel(r.label)}</span></div>
      <div>Confianza: ${conf}</div>
    `;

    list.appendChild(item);
  });

  historyContent.innerHTML = "";
  historyContent.appendChild(list);
}

menuToggle.addEventListener("click", () => {
  menuDropdown.classList.toggle("open");
});

btnVerHistorial.addEventListener("click", () => {
  menuDropdown.classList.remove("open");
  loadHistory();
  historyPanel.classList.remove("hidden");
});

closeHistory.addEventListener("click", () => {
  historyPanel.classList.add("hidden");
});

document.addEventListener("click", (e) => {
  if (!menuDropdown.contains(e.target) && !menuToggle.contains(e.target)) {
    menuDropdown.classList.remove("open");
  }
});

