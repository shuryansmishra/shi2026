import React, { useRef, useState, useEffect } from "react";

export default function Navbar({ activeTab, onTabChange, backendStatus, onOpenLogin, currentUser, onLogout }) {
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

  const firstName = currentUser?.displayName?.split(" ")[0] || currentUser?.email?.split("@")[0] || "Researcher";

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
          {currentUser ? (
            /* Logged in User Capsule with Profile Trigger & Quick Logout */
            <div className="nav-user-cluster">
              <button
                type="button"
                className="nav-user-pill"
                onClick={onOpenLogin}
                title="View Researcher Profile & Cloud Sync"
              >
                <div className="user-pill-avatar">
                  {currentUser.photoURL ? (
                    <img src={currentUser.photoURL} alt="Avatar" className="user-pill-img" />
                  ) : (
                    <span className="user-pill-initial">
                      {firstName[0].toUpperCase()}
                    </span>
                  )}
                </div>
                <span className="user-pill-name">{firstName}</span>
                <span className="user-online-dot"></span>
              </button>

              <button
                type="button"
                className="nav-logout-btn"
                onClick={onLogout}
                title="Sign Out"
                aria-label="Sign Out"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                  <polyline points="16 17 21 12 16 7"></polyline>
                  <line x1="21" y1="12" x2="9" y2="12"></line>
                </svg>
                <span className="logout-text">Logout</span>
              </button>
            </div>
          ) : (
            /* Logged Out: Sign In Button */
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
          )}
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
