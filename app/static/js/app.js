/**
 * Pyronix AI — Satellite Wildfire Intelligence Client Controller
 * Single-Page Reactive Controller (Vanilla JS)
 */

document.addEventListener("DOMContentLoaded", () => {
    // State management
    const state = {
        activePreset: "palisades",
        hasAnalyzed: false,
        analyzedPreset: null,
        currentPreviews: {},
        activeLayer: "probability_map",
        secondaryLayer: "optical_rgb",
        isSplitActive: false,
        sitrepMarkdown: "",
        lastAnalyzedIncident: null,
        isAnalyzing: false
    };

    // DOM Elements - Telemetry Bar
    const valStatus = document.getElementById("val-status");
    const valDevice = document.getElementById("val-device");
    const valModels = document.getElementById("val-models");
    const valIncident = document.getElementById("val-incident");

    // DOM Elements - Presets & Forms
    const presetCards = document.querySelectorAll(".preset-card");
    const uploadForm = document.getElementById("upload-form");
    const inputIncidentName = document.getElementById("input-incident-name");
    const fileOptical = document.getElementById("file-optical");
    const fileSar = document.getElementById("file-sar");
    const nameOptical = document.getElementById("name-optical");
    const nameSar = document.getElementById("name-sar");
    const dropOptical = document.getElementById("dropzone-optical");
    const dropSar = document.getElementById("dropzone-sar");
    const inputThreshold = document.getElementById("input-threshold");
    const valThreshold = document.getElementById("val-threshold");

    // DOM Elements - Situation HUD
    const hudThreatBadge = document.getElementById("hud-threat-badge");
    const hudThreatScore = document.getElementById("hud-threat-score");
    const hudBurnedKm2 = document.getElementById("hud-burned-km2");
    const hudBurnedAcres = document.getElementById("hud-burned-acres");
    const hudPerimeterKm = document.getElementById("hud-perimeter-km");
    const hudClusters = document.getElementById("hud-clusters");
    const hudModelUsed = document.getElementById("hud-model-used");
    const hudArbitrationMode = document.getElementById("hud-arbitration-mode");
    const hudDebrisHazard = document.getElementById("hud-debris-hazard");
    const hudHydroRisk = document.getElementById("hud-hydro-risk");

    // DOM Elements - Viewer
    const primaryImg = document.getElementById("primary-view-img");
    const secondaryImg = document.getElementById("secondary-view-img");
    const splitContainer = document.getElementById("split-container");
    const splitDivider = document.getElementById("split-divider");
    const layerTabs = document.querySelectorAll(".layer-tab");
    const layerOpacity = document.getElementById("layer-opacity");
    const btnToggleSplit = document.getElementById("btn-toggle-split");
    const btnFullscreen = document.getElementById("btn-fullscreen");
    const canvasStage = document.getElementById("canvas-stage");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingStepText = document.getElementById("loading-step-text");
    const legendTitle = document.getElementById("legend-title");
    const legendBar = document.getElementById("legend-bar");
    const legendMin = document.getElementById("legend-min");
    const legendMax = document.getElementById("legend-max");

    // DOM Elements - Telemetry Agent Nodes
    const arbCloud = document.getElementById("arb-cloud");
    const arbSar = document.getElementById("arb-sar");
    const arbModel = document.getElementById("arb-model");
    const delExtent = document.getElementById("del-extent");
    const delCoverage = document.getElementById("del-coverage");
    const delUncertainty = document.getElementById("del-uncertainty");
    const sevG1 = document.getElementById("sev-g1");
    const sevG2 = document.getElementById("sev-g2");
    const sevG3 = document.getElementById("sev-g3");
    const riskDebris = document.getElementById("risk-debris");
    const riskContain = document.getElementById("risk-contain");
    const riskScore = document.getElementById("risk-score");
    const orchState = document.getElementById("orch-state");
    const orchTime = document.getElementById("orch-time");

    // DOM Elements - Chatbot
    const chatThread = document.getElementById("chat-thread");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const promptChips = document.querySelectorAll(".chip-btn");

    // DOM Elements - Modal & Exports
    const sitrepModal = document.getElementById("sitrep-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnViewSitrepModal = document.getElementById("btn-view-sitrep-modal");
    const sitrepMarkdownContent = document.getElementById("sitrep-markdown-content");
    const btnDownloadSitrep = document.getElementById("btn-download-sitrep");
    const btnCopySitrep = document.getElementById("btn-copy-sitrep");
    const btnModalCopy = document.getElementById("btn-modal-copy");
    const btnModalDownload = document.getElementById("btn-modal-download");
    const toastContainer = document.getElementById("toast-container");

    // =========================================================================
    // 1. INITIALIZATION & HARDWARE STATUS
    // =========================================================================
    async function fetchSystemStatus() {
        try {
            const res = await fetch("/api/status");
            if (!res.ok) throw new Error("Status API error");
            const data = await res.json();
            
            valStatus.textContent = data.status || "OPERATIONAL";
            valDevice.textContent = `${data.device_name || 'CUDA'} (${data.vram_gb || 0} GB)`;
            valModels.textContent = `${data.total_models || 5} ResNet-34 Models`;
            if (data.active_incident) {
                valIncident.textContent = data.active_incident;
            }
        } catch (err) {
            console.warn("Could not fetch system status:", err);
            valStatus.textContent = "STANDBY";
            valStatus.parentElement.classList.remove("status-active");
        }
    }

    // Threshold slider live indicator
    inputThreshold.addEventListener("input", (e) => {
        valThreshold.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // File input handlers
    function setupFilePicker(inputEl, displayEl, dropZone) {
        inputEl.addEventListener("change", (e) => {
            if (inputEl.files && inputEl.files[0]) {
                displayEl.textContent = inputEl.files[0].name;
            } else {
                displayEl.textContent = "No file selected";
            }
        });

        // Drag & drop visuals
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                inputEl.files = e.dataTransfer.files;
                displayEl.textContent = e.dataTransfer.files[0].name;
            }
        });
    }

    setupFilePicker(fileOptical, nameOptical, dropOptical);
    setupFilePicker(fileSar, nameSar, dropSar);

    // =========================================================================
    // 2. MISSION TARGET PRESETS & ANALYSIS
    // =========================================================================
    presetCards.forEach(card => {
        card.addEventListener("click", (e) => {
            presetCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            state.activePreset = card.dataset.presetId;
        });

        const runBtn = card.querySelector(".btn-preset-run");
        if (runBtn) {
            runBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                presetCards.forEach(c => c.classList.remove("active"));
                card.classList.add("active");
                state.activePreset = card.dataset.presetId;
                runPresetAnalysis(state.activePreset);
            });
        }
    });

    async function runPresetAnalysis(presetId) {
        if (state.isAnalyzing) return;
        state.isAnalyzing = true;
        showLoadingOverlay();

        try {
            const formData = new FormData();
            formData.append("preset_id", presetId);
            formData.append("threshold", inputThreshold.value);

            // Step status simulation
            setLoadingStep("SensorArbitrator evaluating cloud cover & SAR signal-to-noise ratio...");
            setTimeout(() => {
                setLoadingStep("DelineationAgent routing to ResNet-34 U-Net and computing burned probability...");
            }, 1200);
            setTimeout(() => {
                setLoadingStep("SeverityQuantifierAgent stratifying damage into Copernicus EMS Grades 1-3...");
            }, 2400);
            setTimeout(() => {
                setLoadingStep("RiskAssessmentAgent evaluating debris flow hazard & compiling BAER prescriptions...");
            }, 3600);

            const res = await fetch("/api/analyze/preset", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned code ${res.status}`);
            }

            const data = await res.json();
            handleAnalysisResult(data);
            showToast(`Analysis completed successfully for ${data.incident_name}`, "success");
        } catch (err) {
            console.error("Analysis failed:", err);
            showToast(`Analysis error: ${err.message}`, "error");
        } finally {
            hideLoadingOverlay();
            state.isAnalyzing = false;
        }
    }

    // Custom Upload Form submission
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (state.isAnalyzing) return;

        const optF = fileOptical.files[0];
        const sarF = fileSar.files[0];

        if (!optF && !sarF) {
            showToast("Please select at least one Optical or SAR satellite imagery file.", "error");
            return;
        }

        state.isAnalyzing = true;
        showLoadingOverlay();
        setLoadingStep("Ingesting multi-modal satellite files and extracting radiometric bands...");

        try {
            const formData = new FormData();
            if (optF) formData.append("optical_file", optF);
            if (sarF) formData.append("sar_file", sarF);
            formData.append("incident_name", inputIncidentName.value.trim() || "User Sortie");
            formData.append("threshold", inputThreshold.value);

            const res = await fetch("/api/analyze/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned error ${res.status}`);
            }

            const data = await res.json();
            handleAnalysisResult(data);
            showToast(`Uploaded dataset analyzed successfully!`, "success");
        } catch (err) {
            console.error("Upload analysis failed:", err);
            showToast(`Upload failed: ${err.message}`, "error");
        } finally {
            hideLoadingOverlay();
            state.isAnalyzing = false;
        }
    });

    // =========================================================================
    // 3. UI UPDATE & SITUATION HUD
    // =========================================================================
    function handleAnalysisResult(data) {
        state.hasAnalyzed = true;
        state.analyzedPreset = state.activePreset;
        state.lastAnalyzedIncident = data.incident_name;
        state.currentPreviews = data.previews || {};
        const sitrep = data.sitrep || {};
        state.sitrepMarkdown = sitrep.markdown_report || "No markdown report generated.";

        // Update Header Target
        valIncident.textContent = data.incident_name;

        // Update Situation Quick-HUD
        const threatLevel = data.threat_level || "UNKNOWN";
        hudThreatBadge.textContent = threatLevel;
        hudThreatBadge.className = "threat-badge " + (threatLevel.toLowerCase().includes("crit") ? "threat-critical" : "threat-high");

        const riskData = sitrep.risk || {};
        hudThreatScore.textContent = `${riskData.threat_score || 0}/100`;

        if (data.affected_area_display && data.affected_area_display.includes("unavailable")) {
            hudBurnedKm2.textContent = "N/A";
            hudBurnedAcres.textContent = "(Geospatial area unavailable)";
        } else if (data.burned_area_km2 !== null && data.burned_area_km2 !== undefined && data.burned_area_km2 > 0) {
            hudBurnedKm2.textContent = (data.burned_area_km2).toFixed(2);
            hudBurnedAcres.textContent = `(${(data.burned_area_acres || 0).toFixed(1)} acres)`;
        } else {
            hudBurnedKm2.textContent = "N/A";
            hudBurnedAcres.textContent = "(Geospatial area unavailable)";
        }
        hudPerimeterKm.textContent = (data.perimeter_km || 0).toFixed(2);

        // Ground Truth Validation Status (Scientific Rule)
        const hudGtBadge = document.getElementById("hud-gt-badge");
        const hudGtDesc = document.getElementById("hud-gt-desc");
        if (hudGtBadge && hudGtDesc) {
            const gtStatus = data.ground_truth_status || (data.ground_truth && data.ground_truth.status) || "Available";
            if (gtStatus.toLowerCase().includes("not available") || gtStatus.toLowerCase().includes("unverified")) {
                hudGtBadge.textContent = "NOT AVAILABLE";
                hudGtBadge.className = "hazard-chip hazard-high";
                hudGtDesc.textContent = "Validation Status: Unverified";
            } else {
                hudGtBadge.textContent = "AVAILABLE";
                hudGtBadge.className = "hazard-chip hazard-low";
                hudGtDesc.textContent = (data.ground_truth && data.ground_truth.dataset) || "Verified Benchmark";
            }
        }

        const sevData = sitrep.severity || {};
        hudClusters.textContent = `${sevData.num_fire_clusters || 1} clusters`;

        hudModelUsed.textContent = data.recommended_model || "Dual-Pol ResNet";
        hudArbitrationMode.textContent = data.arbitration_mode || "Optical";

        const debrisData = riskData.debris_flow_hazard || {};
        const dLevel = (debrisData.level || "MODERATE").toUpperCase();
        hudDebrisHazard.textContent = dLevel;
        hudDebrisHazard.className = "hazard-chip " + (dLevel.includes("HIGH") ? "hazard-high" : (dLevel.includes("MOD") ? "hazard-mod" : "hazard-low"));
        hudHydroRisk.textContent = `Hydrophobicity: ${riskData.soil_hydrophobicity_risk || 'Moderate'}`;

        // Update Agent Telemetry Cards
        const arbData = sitrep.arbitration || {};
        arbCloud.textContent = `${(arbData.cloud_cover_pct || 0).toFixed(1)}%`;
        arbSar.textContent = `${(arbData.sar_quality_score || 1.0).toFixed(2)}`;
        arbModel.textContent = arbData.recommended_model || "S2_RGB_only";

        const delData = sitrep.delineation || {};
        delExtent.textContent = `${(delData.burned_area_km2 || 0).toFixed(2)} km²`;
        delCoverage.textContent = `${(delData.burn_percentage || 0).toFixed(1)}%`;
        delUncertainty.textContent = `${(delData.mean_uncertainty || 0.05).toFixed(3)}`;

        const tiers = sevData.severity_tiers || {};
        sevG1.textContent = `${tiers.low_severity?.pct_of_fire || 0}%`;
        sevG2.textContent = `${tiers.moderate_severity?.pct_of_fire || 0}%`;
        sevG3.textContent = `${tiers.high_severity?.pct_of_fire || 0}%`;

        riskDebris.textContent = dLevel;
        riskContain.textContent = riskData.containment_complexity?.level || "Moderate";
        riskScore.textContent = `${riskData.threat_score || 0}/100`;

        orchState.textContent = "Dispatched";
        orchTime.textContent = new Date().toLocaleTimeString();

        // Switch to probability map by default
        selectLayer("probability_map");

        // Inform Chatbot with auto notification
        appendBotMessage(
            `**Analysis Complete for ${data.incident_name}**\n` +
            `• Burned Area: **${data.burned_area_km2.toFixed(2)} km²** (${data.burned_area_acres.toFixed(1)} acres)\n` +
            `• Perimeter: **${data.perimeter_km.toFixed(2)} km** across ${sevData.num_fire_clusters || 1} distinct fire clusters\n` +
            `• Overall Threat: **${threatLevel}** (Hazard Score: ${riskData.threat_score}/100)\n` +
            `• Sensor Arbitration: Model **${data.recommended_model}** selected via **${data.arbitration_mode}** routing.\n\n` +
            `You can now inspect all satellite layers or ask tactical questions.`
        );
    }

    // =========================================================================
    // 4. MULTI-LAYER VIEWER & SPLIT COMPARISON
    // =========================================================================
    layerTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            layerTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            selectLayer(tab.dataset.layer);
        });
    });

    function selectLayer(layerKey) {
        state.activeLayer = layerKey;
        const imgData = state.currentPreviews[layerKey];

        if (imgData) {
            primaryImg.src = imgData;
            primaryImg.style.display = "block";
        } else {
            // Fallback if specific layer not present in this preset
            console.warn(`Layer ${layerKey} not available in previews.`);
            if (state.currentPreviews["probability_map"]) {
                primaryImg.src = state.currentPreviews["probability_map"];
            }
        }

        updateLegend(layerKey);

        // Update secondary image for split comparison
        if (state.isSplitActive) {
            const secKey = (layerKey === "optical_rgb") ? "probability_map" : "optical_rgb";
            if (state.currentPreviews[secKey]) {
                secondaryImg.src = state.currentPreviews[secKey];
            }
        }
    }

    function updateLegend(layerKey) {
        switch(layerKey) {
            case "probability_map":
                legendTitle.textContent = "Burn Probability (ResNet-34)";
                legendMin.textContent = "0.0";
                legendMax.textContent = "1.0";
                legendBar.style.background = "linear-gradient(90deg, #000004 0%, #721f81 35%, #f1605d 70%, #fcfdbf 100%)";
                break;
            case "binary_mask":
                legendTitle.textContent = "Delineated Perimeter (Binary Mask)";
                legendMin.textContent = "Unburned";
                legendMax.textContent = "Burned";
                legendBar.style.background = "linear-gradient(90deg, #1e293b 0%, #ef4444 100%)";
                break;
            case "severity_map":
                legendTitle.textContent = "Copernicus EMS Damage Grades 1-3";
                legendMin.textContent = "Low (G1)";
                legendMax.textContent = "High (G3)";
                legendBar.style.background = "linear-gradient(90deg, #fed976 0%, #fd8d3c 50%, #bd0026 100%)";
                break;
            case "uncertainty_map":
                legendTitle.textContent = "Boundary Uncertainty (Entropy)";
                legendMin.textContent = "Certain (0.0)";
                legendMax.textContent = "Ambiguous (1.0)";
                legendBar.style.background = "linear-gradient(90deg, #440154 0%, #21918c 50%, #fde725 100%)";
                break;
            case "optical_rgb":
                legendTitle.textContent = "Sentinel-2 MSI True Color (RGB)";
                legendMin.textContent = "490nm";
                legendMax.textContent = "665nm";
                legendBar.style.background = "linear-gradient(90deg, #1e3a8a 0%, #059669 50%, #dc2626 100%)";
                break;
            case "sar_vh":
                legendTitle.textContent = "Sentinel-1A SAR Radar (VH Backscatter)";
                legendMin.textContent = "-28 dB";
                legendMax.textContent = "-5 dB";
                legendBar.style.background = "linear-gradient(90deg, #000000 0%, #888888 50%, #ffffff 100%)";
                break;
            case "dnbr":
                legendTitle.textContent = "USGS Differenced NBR (Ground Truth)";
                legendMin.textContent = "-0.2";
                legendMax.textContent = "+0.8";
                legendBar.style.background = "linear-gradient(90deg, #1a9850 0%, #ffffbf 50%, #d73027 100%)";
                break;
            default:
                legendTitle.textContent = "Active Satellite Layer";
                legendMin.textContent = "Min";
                legendMax.textContent = "Max";
                legendBar.style.background = "linear-gradient(90deg, #333 0%, #fff 100%)";
        }
    }

    // Layer Opacity Blend Slider
    layerOpacity.addEventListener("input", (e) => {
        primaryImg.style.opacity = e.target.value;
    });

    // Split-Screen Comparison Tool
    btnToggleSplit.addEventListener("click", () => {
        state.isSplitActive = !state.isSplitActive;
        if (state.isSplitActive) {
            btnToggleSplit.classList.add("active");
            splitContainer.style.display = "block";
            // Set secondary comparison image
            const secKey = (state.activeLayer === "optical_rgb") ? "probability_map" : "optical_rgb";
            if (state.currentPreviews[secKey]) {
                secondaryImg.src = state.currentPreviews[secKey];
            } else if (state.currentPreviews["optical_rgb"]) {
                secondaryImg.src = state.currentPreviews["optical_rgb"];
            }
            updateSplitPosition(50);
            showToast("Comparison Mode: Drag slider to compare layers", "info");
        } else {
            btnToggleSplit.classList.remove("active");
            splitContainer.style.display = "none";
        }
    });

    function updateSplitPosition(pct) {
        pct = Math.max(5, Math.min(95, pct));
        splitDivider.style.left = `${pct}%`;
        splitContainer.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
    }

    let isDraggingSplit = false;
    splitDivider.addEventListener("mousedown", () => { isDraggingSplit = true; });
    window.addEventListener("mouseup", () => { isDraggingSplit = false; });
    window.addEventListener("mousemove", (e) => {
        if (!isDraggingSplit) return;
        const rect = canvasStage.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = (x / rect.width) * 100;
        updateSplitPosition(pct);
    });

    // Fullscreen Toggle
    btnFullscreen.addEventListener("click", () => {
        if (!document.fullscreenElement) {
            canvasStage.requestFullscreen().catch(err => {
                showToast(`Fullscreen error: ${err.message}`, "error");
            });
        } else {
            document.exitFullscreen();
        }
    });

    // Loading overlay handlers
    function showLoadingOverlay() {
        loadingOverlay.style.display = "flex";
    }

    function hideLoadingOverlay() {
        loadingOverlay.style.display = "none";
    }

    function setLoadingStep(text) {
        loadingStepText.textContent = text;
    }

    // =========================================================================
    // 5. TACTICAL OFFICER CHATBOT
    // =========================================================================
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        chatInput.value = "";
        appendUserMessage(query);
        await sendChatQuery(query);
    });

    // Prompt Quick Chips
    promptChips.forEach(chip => {
        chip.addEventListener("click", async () => {
            const query = chip.dataset.query;
            appendUserMessage(query);
            await sendChatQuery(query);
        });
    });

    async function checkChatbotHealth() {
        const badge = document.getElementById("chat-status-badge");
        const greetingHeading = document.getElementById("chat-greeting-heading");
        const greetingBody = document.getElementById("chat-greeting-body");

        try {
            const res = await fetch("/api/chat/health");
            if (!res.ok) throw new Error("Health check unreachable");
            const data = await res.json();

            if (data.configured && data.status === "ok") {
                if (badge) {
                    badge.className = "status-online";
                    badge.innerHTML = `<span class="pulse-dot"></span> GEMINI CONNECTED`;
                }
                if (greetingHeading) greetingHeading.innerHTML = `<strong>Pyronix AI Assistant (${data.model || "Gemini"})</strong>`;
                if (greetingBody) greetingBody.textContent = "Gemini AI assistant connected. Ask about the current satellite analysis, model output, or wildfire methodology.";
            } else {
                if (badge) {
                    badge.className = "status-offline";
                    badge.innerHTML = `GEMINI OFFLINE`;
                }
                if (greetingHeading) greetingHeading.innerHTML = `<strong>AI Assistant Unavailable</strong>`;
                if (greetingBody) greetingBody.textContent = "AI assistant is temporarily unavailable. GEMINI_API_KEY is not configured.";
            }
        } catch (err) {
            if (badge) {
                badge.className = "status-offline";
                badge.innerHTML = `GEMINI OFFLINE`;
            }
            if (greetingHeading) greetingHeading.innerHTML = `<strong>AI Assistant Offline</strong>`;
            if (greetingBody) greetingBody.textContent = "AI assistant is temporarily unavailable.";
        }
    }

    async function sendChatQuery(query) {
        if (!query || query.length === 0) return;
        if (query.length > 1000) {
            showToast("Query exceeds maximum allowed limit of 1000 characters.", "error");
            return;
        }

        const placeholderId = "msg-thinking-" + Date.now();
        appendThinkingMessage(placeholderId);

        try {
            const formData = new FormData();
            formData.append("query", query);
            if (state.hasAnalyzed && state.analyzedPreset) {
                formData.append("preset_id", state.analyzedPreset);
            }

            const res = await fetch("/api/chat", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned code ${res.status}`);
            }

            const data = await res.json();
            removeThinkingMessage(placeholderId);
            appendBotMessage(data.reply, data.sources || ["Gemini AI Assistant"]);
        } catch (err) {
            removeThinkingMessage(placeholderId);
            appendBotMessage(`**Notice:** ${err.message}`);
            showToast(`Chat error: ${err.message}`, "error");
        }
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-message msg-user";
        msgDiv.innerHTML = `
            <div class="msg-avatar user-avatar">YOU</div>
            <div class="msg-bubble"><p>${escapeHtml(text)}</p></div>
        `;
        chatThread.appendChild(msgDiv);
        scrollChatToBottom();
    }

    function appendBotMessage(markdownText, sources = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-message msg-system";

        let html = formatMarkdownToHtml(markdownText);
        let sourcesHtml = "";
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="bot-sources-row">
                    <span class="sources-label">SOURCES:</span>
                    ${sources.map(s => `<span class="source-tag">${s}</span>`).join(" ")}
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="msg-avatar bot-avatar">AI</div>
            <div class="msg-bubble">
                ${html}
                ${sourcesHtml}
            </div>
        `;
        chatThread.appendChild(msgDiv);
        scrollChatToBottom();
    }

    function appendThinkingMessage(id) {
        const msgDiv = document.createElement("div");
        msgDiv.id = id;
        msgDiv.className = "chat-message msg-system";
        msgDiv.innerHTML = `
            <div class="msg-avatar bot-avatar">AI</div>
            <div class="msg-bubble"><p><em>Analyzing multi-agent satellite telemetry with Gemini...</em></p></div>
        `;
        chatThread.appendChild(msgDiv);
        scrollChatToBottom();
    }

    function removeThinkingMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollChatToBottom() {
        chatThread.scrollTop = chatThread.scrollHeight;
    }

    // Helper to format basic markdown to HTML
    function formatMarkdownToHtml(text) {
        let escaped = escapeHtml(text);
        // Bold
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic
        escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Bullet points
        escaped = escaped.replace(/• (.*?)(\n|$)/g, '<li>$1</li>');
        escaped = escaped.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
        // Line breaks
        escaped = escaped.replace(/\n\n/g, '<p></p>');
        escaped = escaped.replace(/\n/g, '<br>');
        return escaped;
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // =========================================================================
    // 6. SITREP MODAL & EXPORT ACTIONS
    // =========================================================================
    function openSitrepModal() {
        if (!state.sitrepMarkdown) {
            showToast("No active Situation Report available. Run an analysis first.", "info");
            return;
        }
        sitrepMarkdownContent.textContent = state.sitrepMarkdown;
        sitrepModal.style.display = "flex";
    }

    function closeSitrepModal() {
        sitrepModal.style.display = "none";
    }

    btnViewSitrepModal.addEventListener("click", openSitrepModal);
    btnCloseModal.addEventListener("click", closeSitrepModal);
    sitrepModal.addEventListener("click", (e) => {
        if (e.target === sitrepModal) closeSitrepModal();
    });

    function downloadMarkdownReport() {
        if (!state.sitrepMarkdown) {
            showToast("No Situation Report available to download.", "error");
            return;
        }
        const blob = new Blob([state.sitrepMarkdown], { type: "text/markdown;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const filename = `${(state.lastAnalyzedIncident || "wildfire").toLowerCase().replace(/[^a-z0-9]/g, "_")}_sitrep.md`;
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast(`Downloaded ${filename}`, "success");
    }

    async function copyReportToClipboard() {
        if (!state.sitrepMarkdown) {
            showToast("No Situation Report available to copy.", "error");
            return;
        }
        try {
            await navigator.clipboard.writeText(state.sitrepMarkdown);
            showToast("Situation Report copied to clipboard!", "success");
        } catch (err) {
            showToast("Failed to copy report to clipboard.", "error");
        }
    }

    btnDownloadSitrep.addEventListener("click", downloadMarkdownReport);
    btnCopySitrep.addEventListener("click", copyReportToClipboard);
    btnModalDownload.addEventListener("click", downloadMarkdownReport);
    btnModalCopy.addEventListener("click", copyReportToClipboard);

    // Toast utility
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // =========================================================================
    // 7. INITIAL BOOTSTRAP
    // =========================================================================
    fetchSystemStatus();
    checkChatbotHealth();

    // Auto-load Pacific Palisades default analysis so user immediately sees rich telemetry
    setTimeout(() => {
        runPresetAnalysis("palisades");
    }, 600);
});
