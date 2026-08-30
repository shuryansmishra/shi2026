import React, { useEffect, useRef } from "react";
import lottie from "lottie-web/build/player/lottie_light";
import loginAnimationData from "../assets/Login.json";

export default function LoginModal({ isOpen, onClose }) {
  const animContainerRef = useRef(null);

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [isOpen, onClose]);

  useEffect(() => {
    let anim = null;
    if (isOpen && animContainerRef.current) {
      anim = lottie.loadAnimation({
        container: animContainerRef.current,
        renderer: "svg",
        loop: true,
        autoplay: true,
        animationData: loginAnimationData,
      });
    }
    return () => {
      if (anim) anim.destroy();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="login-modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
          ✕
        </button>

        {/* Lottie Animation Header */}
        <div className="login-animation-wrap">
          <div
            ref={animContainerRef}
            style={{ width: "200px", height: "200px", margin: "0 auto" }}
          />
        </div>

        {/* Modal Header & Beta Prototype Notice */}
        <div className="login-modal-body">
          <div className="prototype-badge">
            <span className="pulse-dot"></span>
            <span>PROTOTYPE PHASE &bull; IN ACTIVE DEVELOPMENT</span>
          </div>

          <h2 className="login-modal-title">Portal Authentication &amp; Accounts</h2>

          <p className="login-modal-desc">
            We are actively working on this feature! The SatQuery AI portal and SSO login
            infrastructure are currently in the <strong>prototype phase</strong>.
          </p>

          <div className="prototype-info-box">
            <div className="info-icon">⚡</div>
            <div className="info-text">
              <strong>Coming Soon:</strong> Personal satellite query logs, custom GeoTIFF workspace
              storage, and secure ISRO multi-sensor access will be available in the upcoming release.
            </div>
          </div>

          <div className="login-modal-actions">
            <button className="login-confirm-btn" onClick={onClose}>
              Got It &bull; Continue Exploring
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
