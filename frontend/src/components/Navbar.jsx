import React, { useRef, useState, useEffect } from "react";

export default function Navbar({ activeTab, onTabChange, backendStatus, onOpenLogin }) {
  const tabsRef = useRef({});
  const [sliderStyle, setSliderStyle] = useState({ left: 0, width: 0, opacity: 0 });

  useEffect(() => {
    function updateSlider() {
      const el = tabsRef.current[activeTab];
      if (el) {
        setSliderStyle({
          left: el.offsetLeft,
          width: el.offsetWidth,
          opacity: 1,
        });
      }
    }

    updateSlider();
    window.addEventListener("resize", updateSlider);
    return () => window.removeEventListener("resize", updateSlider);
  }, [activeTab]);

  return (
    <header className="navbar">
      <div className="nav-top-bar">
        <div className="nav-left">
          <div className="title-group">
            <div className="brand-frame">
              <h1 className="brand-title">ISRO SatQuery AI</h1>
            </div>
          </div>
        </div>

        <div className="nav-right">
          <button className="nav-login-btn" onClick={onOpenLogin} aria-label="Portal Login">
            <span className="login-shimmer" />
            <div className="login-user-icon-wrap">
              <svg className="login-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
            </div>
            <span className="login-btn-text">Sign In</span>
            <svg className="login-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
          </button>
        </div>
      </div>

      <nav className="nav-tabs" aria-label="Portal modes">
        <div
          className="nav-tab-slider"
          style={{
            transform: `translateX(${sliderStyle.left}px)`,
            width: `${sliderStyle.width}px`,
            opacity: sliderStyle.opacity,
          }}
        />
        <button
          ref={(el) => (tabsRef.current["single_vqa"] = el)}
          className={`nav-tab ${activeTab === "single_vqa" ? "active" : ""}`}
          onClick={() => onTabChange("single_vqa")}
        >
          <span className="tab-full">Single Image VQA</span>
          <span className="tab-short">Single VQA</span>
        </button>
        <button
          ref={(el) => (tabsRef.current["change_detection"] = el)}
          className={`nav-tab ${activeTab === "change_detection" ? "active" : ""}`}
          onClick={() => onTabChange("change_detection")}
        >
          <span className="tab-full">Change Detection</span>
          <span className="tab-short">Change Detection</span>
        </button>
        <button
          ref={(el) => (tabsRef.current["live_map"] = el)}
          className={`nav-tab ${activeTab === "live_map" ? "active" : ""}`}
          onClick={() => onTabChange("live_map")}
        >
          <span className="tab-full">Live Map Selection</span>
          <span className="tab-short">Live Map</span>
        </button>
      </nav>
    </header>
  );
}

