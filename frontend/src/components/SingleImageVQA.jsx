import React, { useState, useRef, useEffect } from "react";
import mapboxgl from "mapbox-gl";
import { runQuery, runLocationQuery, createDemoFile } from "../api.js";

const PRESET_VQA_LOCATIONS = [
  { name: "Delhi NCR Urban Sector", short: "Delhi NCR", lat: 28.6139, lng: 77.2090, zoom: 12.5, category: "urban" },
  { name: "Mumbai Coastal Peninsula", short: "Mumbai", lat: 19.0760, lng: 72.8777, zoom: 12.5, category: "urban" },
  { name: "Bengaluru Tech Corridor", short: "Bengaluru", lat: 12.9716, lng: 77.5946, zoom: 13, category: "urban" },
  { name: "Chilika Lake Wetland", short: "Chilika Lake", lat: 19.7165, lng: 85.3218, zoom: 11, category: "water" },
  { name: "Punjab Agricultural Belt", short: "Punjab Agriland", lat: 30.9010, lng: 75.8573, zoom: 12, category: "agriculture" },
  { name: "Hardoi Farmland District", short: "Hardoi UP", lat: 27.3828, lng: 80.1287, zoom: 12.5, category: "agriculture" },
];

export default function SingleImageVQA({ mapboxToken, onShowToast }) {
  const [viewMode, setViewMode] = useState("map"); // "map" (default Mapbox) or "image" (uploaded raw file)
  const [selectedLoc, setSelectedLoc] = useState(PRESET_VQA_LOCATIONS[0]);
  const [searchQuery, setSearchQuery] = useState("");
  const [latInput, setLatInput] = useState(PRESET_VQA_LOCATIONS[0].lat.toFixed(4));
  const [lngInput, setLngInput] = useState(PRESET_VQA_LOCATIONS[0].lng.toFixed(4));
  const [currentCoords, setCurrentCoords] = useState({ lat: PRESET_VQA_LOCATIONS[0].lat, lng: PRESET_VQA_LOCATIONS[0].lng });

  const [scale, setScale] = useState(1);
  const [opacity, setOpacity] = useState(100);
  const [chatInput, setChatInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(3);
  const [confidenceScore, setConfidenceScore] = useState("94.2%");
  const [detectedBBox, setDetectedBBox] = useState(null);

  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "user",
      text: "How many buildings and roads are visible in this satellite scene?",
      time: "10:12 AM",
    },
    {
      id: 2,
      type: "ai",
      text: "Detected 14 residential building footprints and 3 connecting paved roadways with optimal confidence.",
      details: ["Buildings: 14", "Roadways: 3", "Confidence: 94.2%"],
      time: "10:12 AM",
      sender: "SatQuery Core",
    },
  ]);

  const defaultSteps = [
    { title: "Satellite Map Ingest", desc: "Loaded high-res Sentinel-2 RGB composite & coordinate bounds", component: "ingestion.preprocessing" },
    { title: "Scene Classification", desc: "Identified urban settlement pattern, residential zone", component: "LangGraphRouter" },
    { title: "Object Detection", desc: "Detected 14 buildings, 3 roads, 2 water bodies", component: "SingleImageEngine" },
    { title: "VQA Response Generation", desc: "Generated natural language answer with confidence scores", component: "LLMSynthesis" },
  ];

  const [traceSteps, setTraceSteps] = useState(defaultSteps);
  const chatFeedRef = useRef(null);

  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);

  // Initialize Mapbox map
  useEffect(() => {
    if (viewMode !== "map" || !mapContainerRef.current) return;
    if (!mapboxToken) return;

    mapboxgl.accessToken = mapboxToken;

    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: "mapbox://styles/mapbox/satellite-streets-v12",
      center: [selectedLoc.lng, selectedLoc.lat],
      zoom: selectedLoc.zoom || 12,
      attributionControl: false,
    });

    mapInstanceRef.current = map;

    map.on("load", () => {
      // Add custom styled pin marker
      const el = document.createElement("div");
      el.className = "custom-map-marker";
      el.innerHTML = `
        <span class="marker-pulse"></span>
        <span class="marker-dot"></span>
        <span class="marker-title">${selectedLoc.short}</span>
      `;
      markerRef.current = new mapboxgl.Marker(el)
        .setLngLat([selectedLoc.lng, selectedLoc.lat])
        .addTo(map);
    });

    map.on("move", () => {
      const center = map.getCenter();
      setCurrentCoords({ lat: center.lat, lng: center.lng });
    });

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [viewMode, mapboxToken]);

  useEffect(() => {
    if (chatFeedRef.current) {
      chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
    }
  }, [messages, loading]);

  function flyToLoc(loc) {
    setSelectedLoc(loc);
    setLatInput(loc.lat.toFixed(4));
    setLngInput(loc.lng.toFixed(4));
    setCurrentCoords({ lat: loc.lat, lng: loc.lng });
    setViewMode("map");

    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo({
        center: [loc.lng, loc.lat],
        zoom: loc.zoom || 12.5,
        essential: true,
        duration: 1600,
      });

      if (markerRef.current) {
        markerRef.current.setLngLat([loc.lng, loc.lat]);
        const titleEl = markerRef.current.getElement().querySelector(".marker-title");
        if (titleEl) titleEl.innerText = loc.short;
      }
    }
    onShowToast(`Focused satellite scene: ${loc.short}`);
  }

  async function handleSearchLocation(e) {
    if (e) e.preventDefault();
    const query = searchQuery.trim();
    if (!query) return;

    // Check presets first
    const matched = PRESET_VQA_LOCATIONS.find(
      (p) => p.name.toLowerCase().includes(query.toLowerCase()) || p.short.toLowerCase().includes(query.toLowerCase())
    );

    if (matched) {
      flyToLoc(matched);
      setSearchQuery("");
      return;
    }

    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`, {
        headers: { "User-Agent": "SatQuery-AI-VQA/1.0" },
      });
      const data = await res.json();
      if (data && data.length > 0) {
        const item = data[0];
        const newLoc = {
          name: item.display_name,
          short: item.name || query,
          lat: parseFloat(item.lat),
          lng: parseFloat(item.lon),
          zoom: 12.5,
        };
        flyToLoc(newLoc);
        setSearchQuery("");
      } else {
        onShowToast("Location not found, please refine query");
      }
    } catch (err) {
      onShowToast("Geocoding service unavailable");
    }
  }

  function handleCoordSearch(e) {
    if (e) e.preventDefault();
    const lat = parseFloat(latInput);
    const lng = parseFloat(lngInput);
    if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      onShowToast("Invalid coordinates format.");
      return;
    }

    const customLoc = {
      name: `Coordinates (${lat.toFixed(4)}, ${lng.toFixed(4)})`,
      short: `Target (${lat.toFixed(2)}°, ${lng.toFixed(2)}°)`,
      lat,
      lng,
      zoom: 12.5,
    };
    flyToLoc(customLoc);
  }

  function handleZoomIn() {
    if (viewMode === "map" && mapInstanceRef.current) {
      mapInstanceRef.current.zoomIn();
    } else {
      setScale((prev) => Math.min(2.5, +(prev + 0.15).toFixed(2)));
    }
  }

  function handleZoomOut() {
    if (viewMode === "map" && mapInstanceRef.current) {
      mapInstanceRef.current.zoomOut();
    } else {
      setScale((prev) => Math.max(0.75, +(prev - 0.15).toFixed(2)));
    }
  }

  function handleReset() {
    setScale(1);
    setOpacity(100);
    flyToLoc(PRESET_VQA_LOCATIONS[0]);
    onShowToast("Viewer reset to default view.");
  }

  function handleFileChange(e) {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setViewMode("image");
      onShowToast(`Loaded raw image: ${file.name}`);
    }
  }

  function handleChipClick(text) {
    setChatInput(text);
    handleExecuteQuery(text);
  }

  async function handleExecuteQuery(userText) {
    const text = userText || chatInput.trim();
    if (!text) return;

    const nowStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "user",
        text: text,
        time: nowStr,
      },
    ]);
    setChatInput("");
    setLoading(true);
    setActiveStepIndex(0);

    const liveProgress = [
      { title: "Scene Ingestion & Coordinates", desc: `Aligning target: ${selectedLoc.short} (${currentCoords.lat.toFixed(4)}°N, ${currentCoords.lng.toFixed(4)}°E)...`, component: "ingestion" },
      { title: "LangGraph Intent Classification", desc: `Routing spatial query: "${text}"`, component: "LangGraph" },
      { title: "Neural Vision Feature Engine", desc: "Extracting optical & SAR multi-spectral spatial patterns...", component: "SingleImageEngine" },
      { title: "Evidence & LLM Synthesis", desc: "Calculating bounds, counting features, and synthesizing answer...", component: "LLMSynthesis" },
    ];
    setTraceSteps(liveProgress);

    try {
      let fileToUpload = selectedFile;
      if (!fileToUpload) {
        fileToUpload = await createDemoFile(`${selectedLoc.short.toLowerCase().replace(/[^a-z0-9]/g, "_")}.png`);
      }

      const interval = setInterval(() => {
        setActiveStepIndex((prev) => (prev < 3 ? prev + 1 : prev));
      }, 350);

      let data;
      if (selectedFile) {
        data = await runQuery(text, [fileToUpload], [new Date().toISOString().split("T")[0]]);
      } else {
        data = await runLocationQuery(`${text} (Coordinates: ${currentCoords.lat.toFixed(4)}, ${currentCoords.lng.toFixed(4)})`, selectedLoc.name);
      }
      clearInterval(interval);

      if (data.trace?.steps && data.trace.steps.length > 0) {
        const titleMap = {
          ingest: "Image Ingest & Parse",
          langgraph_intent_classification: "LangGraph Intent Classifier",
          langgraph_input_validation: "Input & Modality Validator",
          langgraph_dispatch: "RL Specialist Dispatcher",
          single_image_inference: "Neural Vision Feature Engine",
          build_evidence: "Evidence & Bounding Box Lock",
          synthesize_answer: "LLM Spatial Synthesis",
        };

        const backendLiveSteps = data.trace.steps.map((s) => ({
          title: titleMap[s.step] || s.step.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
          desc: s.output_summary || "Step completed successfully.",
          component: s.component ? s.component.split(".").pop() : "",
        }));

        setTraceSteps(backendLiveSteps);
        setActiveStepIndex(backendLiveSteps.length - 1);
      } else {
        setActiveStepIndex(3);
      }

      const conf = data.evidence?.confidence != null
        ? `${(data.evidence.confidence * 100).toFixed(1)}%`
        : (data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : "94.2%");
      setConfidenceScore(conf);
      if (data.evidence?.bbox_pixel) {
        setDetectedBBox(data.evidence.bbox_pixel);
      } else {
        setDetectedBBox(null);
      }
      const area = data.evidence?.area_ha
        ? `Area: ${data.evidence.area_ha} ha`
        : (data.area_ha ? `Scope: ${data.area_ha.toFixed(1)} ha` : "Scope: 24.5 ha");

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "ai",
          text: data.answer || `Spatial analysis completed for ${selectedLoc.short}. Model identified key terrain features and confidence metrics.`,
          details: [area, `Confidence: ${conf}`],
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          sender: "SatQuery Core",
        },
      ]);
      onShowToast("VQA Query processed!");
    } catch (err) {
      console.error("Single VQA query error:", err);
      setActiveStepIndex(3);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "ai",
          text: `⚠️ Backend Connection Warning: Could not reach FastAPI model server at http://localhost:8000 (${err.message}).\n\nPlease ensure your backend is running:\ncd backend && python -m uvicorn main:app --reload --port 8000`,
          details: ["Status: Offline", "Port: 8000"],
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          sender: "SatQuery System",
        },
      ]);
      onShowToast("Error connecting to backend server.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="workspace-container">
      <div className="workspace">
        {/* Left Sidebar: Image Analysis Log & Location Search */}
        <aside className="sidebar sidebar-left">
          <div className="left-top">
            <header className="sidebar-header">
              <h2>Image Analysis Log</h2>
              <p>LangGraph spatial reasoner &amp; live Mapbox</p>
              <div className="rule"></div>
            </header>

            {/* Location Search Bar */}
            <div className="sidebar-loc-search">
              <div className="section-label">SEARCH SATELLITE SCENE</div>
              <form onSubmit={handleSearchLocation} className="compact-search-form">
                <div className="input-pill-group">
                  <input
                    type="text"
                    className="loc-search-input"
                    placeholder="Search city, lake, region..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <button type="submit" className="loc-action-btn dark" aria-label="Search Scene">
                    Go
                  </button>
                </div>
              </form>
            </div>

            {/* Quick Location & Scene Presets */}
            <div style={{ marginBottom: "12px" }}>
              <div className="section-label" style={{ marginBottom: "6px" }}>PRESET SCENES</div>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {PRESET_VQA_LOCATIONS.map((loc, idx) => (
                  <button
                    key={idx}
                    className={`chip-btn ${selectedLoc.short === loc.short && viewMode === "map" ? "selected" : ""}`}
                    onClick={() => flyToLoc(loc)}
                  >
                    {loc.short}
                  </button>
                ))}
                <label className="chip-btn upload-btn-wrap" style={{ background: "rgba(0,86,167,0.08)" }}>
                  <span>Upload Image</span>
                  <input type="file" accept="image/*,.tif,.tiff" onChange={handleFileChange} />
                </label>
                {viewMode === "image" && (
                  <button className="chip-btn" onClick={() => setViewMode("map")} style={{ background: "#0056A7", color: "#fff" }}>
                    Live Mapbox
                  </button>
                )}
              </div>
            </div>

            {/* Coordinates Quick Jump */}
            <div className="sidebar-loc-search" style={{ borderTop: "1px solid var(--border-line)", paddingTop: "8px" }}>
              <div className="section-label">COORDINATES INPUT</div>
              <form onSubmit={handleCoordSearch} className="compact-search-form">
                <div className="input-pill-group">
                  <input
                    type="text"
                    className="loc-search-input"
                    placeholder="Lat (28.61)"
                    value={latInput}
                    onChange={(e) => setLatInput(e.target.value)}
                  />
                  <input
                    type="text"
                    className="loc-search-input"
                    placeholder="Lng (77.20)"
                    value={lngInput}
                    onChange={(e) => setLngInput(e.target.value)}
                  />
                  <button type="submit" className="loc-action-btn blue" aria-label="Jump Coordinates">
                    Jump
                  </button>
                </div>
              </form>
            </div>

            {/* Timeline */}
            <section className="timeline" aria-label="Image analysis timeline" style={{ marginTop: "10px" }}>
              {traceSteps.map((step, idx) => {
                const isDone = idx <= activeStepIndex;
                const isCurrent = idx === activeStepIndex && loading;
                const isLast = idx === traceSteps.length - 1;
                return (
                  <article key={idx} className={`step ${isLast ? "last" : ""}`}>
                    <div className="step-indicator">
                      <span className={`step-check ${isDone ? "" : "pending"} ${isCurrent ? "active" : ""}`}>
                        {isDone ? "✓" : "○"}
                      </span>
                      {!isLast && (
                        <i className={`step-line ${isDone ? "" : "pending"}`}></i>
                      )}
                    </div>
                    <div className="step-body">
                      <h3>
                        <span>{step.title}</span>
                        {step.component && <span className="step-component-tag">{step.component}</span>}
                      </h3>
                      <p>{step.desc}</p>
                    </div>
                  </article>
                );
              })}
            </section>
          </div>

          <section className="confidence-card">
            <div className="confidence-head">
              <span>ANALYSIS CONFIDENCE</span>
              <span style={{ color: "var(--cyan)", fontSize: "14px" }}>◉</span>
            </div>
            <div className="confidence-value">
              <strong>{confidenceScore}</strong>
              <span>Optimal</span>
            </div>
          </section>
        </aside>

        {/* Center Workspace: Default High-Res Mapbox Satellite Map */}
        <section className="main-map">
          <div className="map-frame">
            {viewMode === "map" ? (
              <div
                ref={mapContainerRef}
                className="vqa-mapbox-canvas"
                style={{ opacity: opacity / 100, transition: "opacity 0.15s ease" }}
              />
            ) : (
              <div className="map-container-full" style={{ position: "relative", overflow: "hidden" }}>
                <img
                  src={previewUrl}
                  alt="Uploaded Satellite Scene"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    transform: `scale(${scale})`,
                    transformOrigin: "center center",
                    opacity: opacity / 100,
                    transition: "transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.15s ease",
                  }}
                />
                {detectedBBox && detectedBBox.length === 4 && (
                  <div
                    style={{
                      position: "absolute",
                      left: `${Math.min(detectedBBox[0], detectedBBox[2]) * 100}%`,
                      top: `${Math.min(detectedBBox[1], detectedBBox[3]) * 100}%`,
                      width: `${Math.abs(detectedBBox[2] - detectedBBox[0]) * 100}%`,
                      height: `${Math.abs(detectedBBox[3] - detectedBBox[1]) * 100}%`,
                      border: "2px solid #00f2fe",
                      boxShadow: "0 0 12px rgba(0, 242, 254, 0.6)",
                      backgroundColor: "rgba(0, 242, 254, 0.15)",
                      pointerEvents: "none",
                      zIndex: 10,
                      borderRadius: "4px",
                      transition: "all 0.3s ease"
                    }}
                  >
                    <span
                      style={{
                        position: "absolute",
                        top: "-22px",
                        left: "0",
                        backgroundColor: "#00f2fe",
                        color: "#0a192f",
                        fontSize: "10px",
                        fontWeight: "700",
                        padding: "2px 6px",
                        borderRadius: "3px",
                        whiteSpace: "nowrap"
                      }}
                    >
                      Qwen Detected Region
                    </span>
                  </div>
                )}
              </div>
            )}

            <div className="draw-tool-badge">
              <span className="live-dot-pulse"></span>
              <span>
                {viewMode === "map"
                  ? `Live Satellite • ${selectedLoc.short}`
                  : (selectedFile ? selectedFile.name : "Custom GeoTIFF")}
              </span>
            </div>

            <div className="coordinates-badge">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ width: "12px", height: "12px", color: "rgba(255,255,255,0.85)" }}>
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
              </svg>
              <span>Lat: {currentCoords.lat.toFixed(4)}° N, Long: {currentCoords.lng.toFixed(4)}° E</span>
            </div>

            {/* Floating Map Controls */}
            <div className="floating-controls">
              <div className="zoom-group">
                <button className="zoom-btn" onClick={handleZoomIn} aria-label="Zoom in">+</button>
                <button className="zoom-btn" onClick={handleZoomOut} aria-label="Zoom out">−</button>
              </div>
              <div className="control-divider"></div>
              <div className="slider-control">
                <span className="slider-label">Layer Opacity:</span>
                <input
                  type="range"
                  className="opacity-slider"
                  min="0"
                  max="100"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  style={{ "--slider-fill": `${opacity}%` }}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Right Sidebar: AI Assistant Chat */}
        <aside className="sidebar sidebar-right">
          <div className="right-content">
            <header className="chat-head">
              <div className="ai-title">
                <span className="dot"></span>
                <h1>AI Assistant</h1>
                <button id="reset" onClick={() => {
                  setMessages([messages[0], messages[1]]);
                  handleReset();
                }}>
                  RESET
                </button>
              </div>
              <div className="rule"></div>
            </header>

            {/* Chat Feed */}
            <section className="chat-feed" ref={chatFeedRef}>
              {messages.map((msg) => (
                <div key={msg.id} className={`bubble-row ${msg.type === "user" ? "user" : "ai"}`}>
                  <div className={`bubble ${msg.type === "user" ? "user" : "ai"}`}>
                    <p>{msg.text}</p>
                    {msg.details && (
                      <div className="bubble-details">
                        {msg.details.map((detail, dIdx) => (
                          <span key={dIdx}>{detail}</span>
                        ))}
                      </div>
                    )}
                    <small>{msg.time} • {msg.sender || "You"}</small>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="bubble-row ai">
                  <div className="bubble ai" style={{ opacity: 0.85 }}>
                    <p>Analyzing satellite telemetry for {selectedLoc.short}...</p>
                    <small>Synthesizing spatial reasoner...</small>
                  </div>
                </div>
              )}
            </section>

            {/* Suggested Action Chips */}
            <div className="actions" style={{ marginTop: "auto" }}>
              <h2>SUGGESTED ACTIONS</h2>
              <div className="chips">
                <button className="chip-btn" onClick={() => handleChipClick("Count Buildings & Structures")}>
                  Count Buildings
                </button>
                <button className="chip-btn" onClick={() => handleChipClick("Detect Water Reservoirs")}>
                  Detect Water
                </button>
                <button className="chip-btn" onClick={() => handleChipClick("Classify Vegetation & Crop Health")}>
                  Classify Land Use
                </button>
                <button className="chip-btn" onClick={() => handleChipClick("Analyze Infrastructure Density")}>
                  Infrastructure Density
                </button>
              </div>
            </div>

            {/* Chat Input Bar */}
            <form
              className="chat-input-bar"
              onSubmit={(e) => {
                e.preventDefault();
                handleExecuteQuery();
              }}
            >
              <input
                type="text"
                className="chat-input-field"
                placeholder={`Ask about ${selectedLoc.short}...`}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={loading}
              />
              <button
                type="submit"
                className="send-icon-btn"
                disabled={loading || !chatInput.trim()}
                aria-label="Send message"
              >
                <svg className="arrow-up" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="19" x2="12" y2="5"></line>
                  <polyline points="5 12 12 5 19 12"></polyline>
                </svg>
              </button>
            </form>

            <div className="beta-disclaimer">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
              </svg>
              <span>Beta Notice: ML models are in active development; inferences may occasionally be inaccurate.</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
