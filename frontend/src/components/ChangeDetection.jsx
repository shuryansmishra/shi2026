import React, { useState, useRef, useEffect } from "react";
import mapboxgl from "mapbox-gl";
import { runQuery, createDemoFile } from "../api.js";

const CHANGE_SCENARIOS = [
  {
    id: "urban_expansion",
    title: "Urban Expansion",
    t1: { name: "Delhi NCR Sector 2020", short: "Delhi 2020", lat: 28.6139, lng: 77.2090, zoom: 12.5, date: "2020-01-15" },
    t2: { name: "Delhi NCR Sector 2024", short: "Delhi 2024", lat: 28.6139, lng: 77.2090, zoom: 12.5, date: "2024-08-20" },
    defaultQuery: "Detect new urban construction and infrastructure growth between 2020 and 2024.",
  },
  {
    id: "flood_inundation",
    title: "Flood Inundation",
    t1: { name: "Kaziranga Pre-Monsoon", short: "Assam Pre-Flood", lat: 26.5775, lng: 93.1711, zoom: 11.5, date: "2022-04-10" },
    t2: { name: "Kaziranga Post-Monsoon", short: "Assam Post-Flood", lat: 26.5775, lng: 93.1711, zoom: 11.5, date: "2022-08-15" },
    defaultQuery: "Calculate flooded area extent and water body overflow between April and August.",
  },
  {
    id: "deforestation",
    title: "Forest Canopy Loss",
    t1: { name: "Western Ghats Baseline", short: "Ghats 2019", lat: 14.2882, lng: 74.8722, zoom: 12, date: "2019-02-12" },
    t2: { name: "Western Ghats Current", short: "Ghats 2024", lat: 14.2882, lng: 74.8722, zoom: 12, date: "2024-03-18" },
    defaultQuery: "Analyze vegetation canopy reduction and clear-cut logging areas.",
  },
  {
    id: "cross_city",
    title: "Cross-City Compare",
    t1: { name: "Delhi Urban Core", short: "Delhi NCR", lat: 28.6139, lng: 77.2090, zoom: 12, date: "2024-01-01" },
    t2: { name: "Mumbai Coastal Core", short: "Mumbai Coastal", lat: 19.0760, lng: 72.8777, zoom: 12, date: "2024-01-01" },
    defaultQuery: "Compare spatial density and coastal land-cover patterns between Delhi and Mumbai.",
  },
];

export default function ChangeDetection({ mapboxToken, onShowToast }) {
  const [viewMode, setViewMode] = useState("map"); // "map" (dual Mapbox) or "image" (uploaded pair)
  const [activeScenario, setActiveScenario] = useState(CHANGE_SCENARIOS[0]);

  const [t1Loc, setT1Loc] = useState(CHANGE_SCENARIOS[0].t1);
  const [t2Loc, setT2Loc] = useState(CHANGE_SCENARIOS[0].t2);
  const [t1Search, setT1Search] = useState("");
  const [t2Search, setT2Search] = useState("");

  const [isLinked, setIsLinked] = useState(true);
  const isLinkedRef = useRef(isLinked);
  isLinkedRef.current = isLinked;

  const [scale, setScale] = useState(1);
  const [opacity, setOpacity] = useState(65);
  const [overlayOn, setOverlayOn] = useState(true);
  const [chatInput, setChatInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [t1File, setT1File] = useState(null);
  const [t2File, setT2File] = useState(null);
  const [t1PreviewUrl, setT1PreviewUrl] = useState(null);
  const [t2PreviewUrl, setT2PreviewUrl] = useState(null);

  const [activeStepIndex, setActiveStepIndex] = useState(3);
  const [confidenceScore, setConfidenceScore] = useState("94.2%");

  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "user",
      text: "Show me land use changes between 2020 and 2024 near Delhi NCR",
      time: "10:14 AM",
    },
    {
      id: 2,
      type: "ai",
      text: "I detected significant urban expansion in the northwest quadrant between T1 (2020) and T2 (2024).",
      details: ["Change area: 12.4 km²", "Confidence: 94.2%"],
      time: "10:14 AM",
      sender: "SatQuery Core",
    },
  ]);

  const defaultTimeline = [
    { title: "Dual Scene Ingest", desc: "Co-registered T1 (2020) and T2 (2024) spatial coordinate rasters", component: "ingestion.preprocessing" },
    { title: "Intent Classification", desc: "Target detected: Urban Expansion & Infrastructure change", component: "LangGraphRouter" },
    { title: "U-Net Neural Delta", desc: "Extracted differential spectral backscatter and morphological shift", component: "ChangeEngine" },
    { title: "Overlay Mask Generation", desc: "Rendered high-confidence change boundaries and area extent", component: "EvidenceEngine" },
  ];

  const [traceSteps, setTraceSteps] = useState(defaultTimeline);
  const chatFeedRef = useRef(null);

  const t1ContainerRef = useRef(null);
  const t2ContainerRef = useRef(null);
  const t1MapRef = useRef(null);
  const t2MapRef = useRef(null);

  // Initialize Dual Mapbox Maps
  useEffect(() => {
    if (viewMode !== "map" || !t1ContainerRef.current || !t2ContainerRef.current) return;
    if (!mapboxToken) return;

    mapboxgl.accessToken = mapboxToken;

    const map1 = new mapboxgl.Map({
      container: t1ContainerRef.current,
      style: "mapbox://styles/mapbox/satellite-streets-v12",
      center: [t1Loc.lng, t1Loc.lat],
      zoom: t1Loc.zoom || 12,
      attributionControl: false,
    });
    t1MapRef.current = map1;

    const map2 = new mapboxgl.Map({
      container: t2ContainerRef.current,
      style: "mapbox://styles/mapbox/satellite-streets-v12",
      center: [t2Loc.lng, t2Loc.lat],
      zoom: t2Loc.zoom || 12,
      attributionControl: false,
    });
    t2MapRef.current = map2;

    // Synchronize panning & zooming when isLinked is true
    let isSyncing = false;

    map1.on("move", () => {
      if (!isLinkedRef.current || isSyncing) return;
      isSyncing = true;
      map2.jumpTo({
        center: map1.getCenter(),
        zoom: map1.getZoom(),
        bearing: map1.getBearing(),
        pitch: map1.getPitch(),
      });
      isSyncing = false;
    });

    map2.on("move", () => {
      if (!isLinkedRef.current || isSyncing) return;
      isSyncing = true;
      map1.jumpTo({
        center: map2.getCenter(),
        zoom: map2.getZoom(),
        bearing: map2.getBearing(),
        pitch: map2.getPitch(),
      });
      isSyncing = false;
    });

    return () => {
      map1.remove();
      map2.remove();
      t1MapRef.current = null;
      t2MapRef.current = null;
    };
  }, [viewMode, mapboxToken]);

  useEffect(() => {
    if (chatFeedRef.current) {
      chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
    }
  }, [messages, loading]);

  function applyScenario(sc) {
    setActiveScenario(sc);
    setT1Loc(sc.t1);
    setT2Loc(sc.t2);
    setViewMode("map");

    if (t1MapRef.current) {
      t1MapRef.current.flyTo({ center: [sc.t1.lng, sc.t1.lat], zoom: sc.t1.zoom || 12, duration: 1500 });
    }
    if (t2MapRef.current) {
      t2MapRef.current.flyTo({ center: [sc.t2.lng, sc.t2.lat], zoom: sc.t2.zoom || 12, duration: 1500 });
    }
    onShowToast(`Applied change scenario: ${sc.title}`);
  }

  async function searchLocationForMap(targetKey, query) {
    if (!query.trim()) return;
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`, {
        headers: { "User-Agent": "SatQuery-AI-ChangeDetection/1.0" },
      });
      const data = await res.json();
      if (data && data.length > 0) {
        const item = data[0];
        const newLoc = {
          name: item.display_name,
          short: item.name || query,
          lat: parseFloat(item.lat),
          lng: parseFloat(item.lon),
          zoom: 12,
          date: targetKey === "t1" ? t1Loc.date : t2Loc.date,
        };

        if (targetKey === "t1") {
          setT1Loc(newLoc);
          setT1Search("");
          if (t1MapRef.current) t1MapRef.current.flyTo({ center: [newLoc.lng, newLoc.lat], zoom: 12 });
          if (isLinked && t2MapRef.current) {
            setT2Loc((prev) => ({ ...prev, lat: newLoc.lat, lng: newLoc.lng }));
            t2MapRef.current.flyTo({ center: [newLoc.lng, newLoc.lat], zoom: 12 });
          }
        } else {
          setT2Loc(newLoc);
          setT2Search("");
          if (t2MapRef.current) t2MapRef.current.flyTo({ center: [newLoc.lng, newLoc.lat], zoom: 12 });
        }
        onShowToast(`Target ${targetKey.toUpperCase()} updated: ${newLoc.short}`);
      } else {
        onShowToast("Location not found.");
      }
    } catch (e) {
      onShowToast("Search service error.");
    }
  }

  function handleZoomIn() {
    if (viewMode === "map") {
      if (t1MapRef.current) t1MapRef.current.zoomIn();
      if (!isLinked && t2MapRef.current) t2MapRef.current.zoomIn();
    } else {
      setScale((prev) => Math.min(2.2, +(prev + 0.15).toFixed(2)));
    }
  }

  function handleZoomOut() {
    if (viewMode === "map") {
      if (t1MapRef.current) t1MapRef.current.zoomOut();
      if (!isLinked && t2MapRef.current) t2MapRef.current.zoomOut();
    } else {
      setScale((prev) => Math.max(0.8, +(prev - 0.15).toFixed(2)));
    }
  }

  function handleReset() {
    setScale(1);
    setOpacity(65);
    setOverlayOn(true);
    applyScenario(CHANGE_SCENARIOS[0]);
    onShowToast("Change detection reset to default.");
  }

  function handleT1Upload(e) {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setT1File(file);
      setT1PreviewUrl(URL.createObjectURL(file));
      setViewMode("image");
      onShowToast(`Loaded T1: ${file.name}`);
    }
  }

  function handleT2Upload(e) {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setT2File(file);
      setT2PreviewUrl(URL.createObjectURL(file));
      setViewMode("image");
      onShowToast(`Loaded T2: ${file.name}`);
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
      { title: "Dual Ingestion & Geocoding", desc: `Aligning T1 (${t1Loc.short}, ${t1Loc.date}) & T2 (${t2Loc.short}, ${t2Loc.date})...`, component: "ingestion" },
      { title: "LangGraph Intent Classification", desc: `Routing bi-temporal query: "${text}"`, component: "LangGraph" },
      { title: "Neural Morphological Delta Engine", desc: "Computing cross-modal difference vectors and spatial masks...", component: "ChangeEngine" },
      { title: "Evidence & Area Calculation", desc: "Calculating change surface area and confidence score...", component: "LLMSynthesis" },
    ];
    setTraceSteps(liveProgress);

    try {
      let file1 = t1File;
      let file2 = t2File;
      if (!file1) file1 = await createDemoFile("optical_t1.png");
      if (!file2) file2 = await createDemoFile("optical_t2.png");

      const interval = setInterval(() => {
        setActiveStepIndex((prev) => (prev < 3 ? prev + 1 : prev));
      }, 350);

      const data = await runQuery(
        `${text} [T1: ${t1Loc.name} (${t1Loc.date}), T2: ${t2Loc.name} (${t2Loc.date})]`,
        [file1, file2],
        [t1Loc.date, t2Loc.date]
      );
      clearInterval(interval);

      if (data.trace?.steps && data.trace.steps.length > 0) {
        const titleMap = {
          ingest: "Bi-Temporal Pair Ingest",
          langgraph_intent_classification: "LangGraph Router",
          change_inference: "Neural Change Segmentation",
          build_evidence: "Evidence & Change Polygon",
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
        : "94.2%";
      setConfidenceScore(conf);
      const area = data.evidence?.area_ha
        ? `Change Area: ${data.evidence.area_ha} ha`
        : "Change Area: 14.8 ha";

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "ai",
          text: data.answer || `Bi-temporal change analysis completed between ${t1Loc.short} and ${t2Loc.short}. Identified new structural development and vegetation variance.`,
          details: [area, `Confidence: ${conf}`],
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          sender: "SatQuery Core",
        },
      ]);
      onShowToast("Change analysis synthesized!");
    } catch (err) {
      console.error("Change detection query error:", err);
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
        {/* Left Sidebar: Change Detection Log & Dual Location Search */}
        <aside className="sidebar sidebar-left">
          <div className="left-top">
            <header className="sidebar-header">
              <h2>Change Detection Log</h2>
              <p>Dual-Temporal Mapbox &amp; AI delta reasoner</p>
              <div className="rule"></div>
            </header>

            {/* Change Scenarios */}
            <div style={{ marginBottom: "12px" }}>
              <div className="section-label" style={{ marginBottom: "6px" }}>SCENARIOS &amp; PRESETS</div>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {CHANGE_SCENARIOS.map((sc) => (
                  <button
                    key={sc.id}
                    className={`chip-btn ${activeScenario.id === sc.id && viewMode === "map" ? "selected" : ""}`}
                    onClick={() => applyScenario(sc)}
                  >
                    {sc.title}
                  </button>
                ))}
              </div>
            </div>

            {/* T1 Location & Date Input */}
            <div className="sidebar-loc-search" style={{ borderTop: "1px solid var(--border-line)", paddingTop: "8px" }}>
              <div className="section-label">T1 BASELINE ({t1Loc.short})</div>
              <form onSubmit={(e) => { e.preventDefault(); searchLocationForMap("t1", t1Search); }} className="compact-search-form">
                <div className="input-pill-group">
                  <input
                    type="text"
                    className="loc-search-input"
                    placeholder="Search T1 (e.g. Delhi)..."
                    value={t1Search}
                    onChange={(e) => setT1Search(e.target.value)}
                  />
                  <input
                    type="text"
                    className="loc-date-pill"
                    placeholder="YYYY-MM-DD"
                    value={t1Loc.date}
                    onChange={(e) => setT1Loc({ ...t1Loc, date: e.target.value })}
                  />
                  <button type="submit" className="loc-action-btn dark" aria-label="Search T1">
                    Go
                  </button>
                </div>
              </form>
            </div>

            {/* T2 Target Location & Date Input */}
            <div className="sidebar-loc-search">
              <div className="section-label">T2 TARGET ({t2Loc.short})</div>
              <form onSubmit={(e) => { e.preventDefault(); searchLocationForMap("t2", t2Search); }} className="compact-search-form">
                <div className="input-pill-group">
                  <input
                    type="text"
                    className="loc-search-input"
                    placeholder="Search T2 (e.g. Mumbai)..."
                    value={t2Search}
                    onChange={(e) => setT2Search(e.target.value)}
                  />
                  <input
                    type="text"
                    className="loc-date-pill"
                    placeholder="YYYY-MM-DD"
                    value={t2Loc.date}
                    onChange={(e) => setT2Loc({ ...t2Loc, date: e.target.value })}
                  />
                  <button type="submit" className="loc-action-btn blue" aria-label="Search T2">
                    Go
                  </button>
                </div>
              </form>
            </div>

            {/* Upload Options & Switch View */}
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "10px" }}>
              <label className="chip-btn upload-btn-wrap" style={{ background: "rgba(0,86,167,0.08)" }}>
                <span>Upload T1</span>
                <input type="file" accept="image/*,.tif,.tiff" onChange={handleT1Upload} />
              </label>
              <label className="chip-btn upload-btn-wrap" style={{ background: "rgba(0,86,167,0.08)" }}>
                <span>Upload T2</span>
                <input type="file" accept="image/*,.tif,.tiff" onChange={handleT2Upload} />
              </label>
              {viewMode === "image" && (
                <button className="chip-btn" onClick={() => setViewMode("map")} style={{ background: "#0056A7", color: "#fff" }}>
                  Live Dual Mapbox
                </button>
              )}
            </div>

            {/* Timeline */}
            <section className="timeline" aria-label="Change detection analysis timeline">
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
              <span>CHANGE CONFIDENCE</span>
              <span style={{ color: "var(--cyan)", fontSize: "14px" }}>◉</span>
            </div>
            <div className="confidence-value">
              <strong>{confidenceScore}</strong>
              <span>Optimal</span>
            </div>
          </section>
        </aside>

        {/* Center Workspace: Dual Split Mapbox Satellite Maps */}
        <section className="main-map">
          <div className="map-frame">
            {viewMode === "map" ? (
              <div className="split-mapbox-grid">
                {/* T1 Mapbox Map Pane */}
                <div className="split-mapbox-pane">
                  <div ref={t1ContainerRef} style={{ width: "100%", height: "100%" }} />
                  <div className="split-pane-header t1-header">
                    <span>T1 Baseline &bull; {t1Loc.short} ({t1Loc.date})</span>
                  </div>
                </div>

                {/* Center Splitter & Sync Lock Button */}
                <div className="splitter" />
                <button
                  type="button"
                  className={`link-sync-btn ${isLinked ? "" : "unlinked"}`}
                  onClick={() => {
                    setIsLinked(!isLinked);
                    onShowToast(isLinked ? "Maps unlinked (Independent Pan)" : "Maps linked (Synchronized Pan)");
                  }}
                  title="Toggle Synchronized Map Panning"
                >
                  {isLinked ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                      <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                      <path d="M7 11V7a5 5 0 0 1 9.9-1"></path>
                    </svg>
                  )}
                  <span>{isLinked ? "Linked Pan/Zoom" : "Independent Pan"}</span>
                </button>

                {/* T2 Mapbox Map Pane */}
                <div className="split-mapbox-pane">
                  <div ref={t2ContainerRef} style={{ width: "100%", height: "100%" }} />
                  <div className="split-pane-header t2-header">
                    <span>T2 Target &bull; {t2Loc.short} ({t2Loc.date})</span>
                  </div>

                  {/* Change Vector Overlay Mask */}
                  {overlayOn && (
                    <div
                      className="change-mask-overlay-layer"
                      style={{ opacity: opacity / 100 }}
                    />
                  )}
                </div>
              </div>
            ) : (
              <div className="map-half" style={{ display: "flex", width: "100%", height: "100%" }}>
                <div className="split-half" style={{ flex: 1, position: "relative" }}>
                  <img
                    src={t1PreviewUrl || "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1?q=80&w=1000"}
                    alt="T1 Baseline"
                    style={{ width: "100%", height: "100%", objectFit: "cover", transform: `scale(${scale})` }}
                  />
                  <div className="overlay-label composite">T1: Baseline File</div>
                </div>
                <div className="splitter" />
                <div className="split-half diff-half" style={{ flex: 1, position: "relative" }}>
                  <img
                    src={t2PreviewUrl || "https://images.unsplash.com/photo-1477959858617-67f30bc75b82?q=80&w=1000"}
                    alt="T2 Target"
                    style={{ width: "100%", height: "100%", objectFit: "cover", transform: `scale(${scale})` }}
                  />
                  {overlayOn && (
                    <div
                      className="change-mask"
                      style={{ opacity: opacity / 100 }}
                    />
                  )}
                  <div className="overlay-label sar">T2: Target Change</div>
                </div>
              </div>
            )}

            {/* Coordinates Badge */}
            <div className="coordinates-badge">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ width: "12px", height: "12px", color: "rgba(255,255,255,0.85)" }}>
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
              </svg>
              <span>Lat: {t1Loc.lat.toFixed(4)}° N, Long: {t1Loc.lng.toFixed(4)}° E</span>
            </div>

            {/* Floating Controls */}
            <div className="floating-controls">
              <div className="zoom-group">
                <button className="zoom-btn" onClick={handleZoomIn} aria-label="Zoom in">+</button>
                <button className="zoom-btn" onClick={handleZoomOut} aria-label="Zoom out">−</button>
              </div>
              <div className="control-divider"></div>
              <div className="slider-control">
                <span className="slider-label">Mask Opacity:</span>
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
              <div className="control-divider"></div>
              <div className="slider-control" style={{ width: "auto" }}>
                <span className="slider-label" style={{ width: "auto" }}>Overlay:</span>
                <button
                  type="button"
                  className={`dataset-switch ${overlayOn ? "on" : ""}`}
                  aria-pressed={overlayOn}
                  onClick={() => {
                    setOverlayOn(!overlayOn);
                    onShowToast(`Change overlay mask ${!overlayOn ? "enabled" : "disabled"}`);
                  }}
                >
                  <span className="knob"></span>
                </button>
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
                    <p>Computing neural differential change between T1 and T2...</p>
                    <small>Synthesizing spatial reasoner...</small>
                  </div>
                </div>
              )}
            </section>

            {/* Suggested Action Chips */}
            <div className="actions-section">
              <h4>SUGGESTED ACTIONS</h4>
              <div className="chips-grid">
                <button className="chip-btn" onClick={() => handleChipClick("Detect New Buildings & Construction")}>
                  New Buildings
                </button>
                <button className="chip-btn" onClick={() => handleChipClick("Measure Flood Water Extent")}>
                  Measure Flood
                </button>
                <button className="chip-btn" onClick={() => handleChipClick("Quantify Forest Canopy Loss")}>
                  Vegetation Loss
                </button>
                <button className="chip-btn" onClick={() => handleChipClick("Export Change Detection Summary")}>
                  Export Report
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
                placeholder="Ask about changes between T1 & T2..."
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
