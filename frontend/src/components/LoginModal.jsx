import React, { useEffect, useRef, useState } from "react";
import lottie from "lottie-web/build/player/lottie_light";
import loginAnimationData from "../assets/Login.json";
import { loginWithGoogle, loginWithGithub, logoutUser } from "../firebase.js";

export default function LoginModal({ isOpen, onClose, currentUser, onShowToast }) {
  const animContainerRef = useRef(null);
  const [loadingProvider, setLoadingProvider] = useState(null);
  const [authError, setAuthError] = useState("");

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
      setAuthError("");
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

  async function handleGoogleLogin() {
    setLoadingProvider("google");
    setAuthError("");
    try {
      const user = await loginWithGoogle();
      if (onShowToast) {
        onShowToast(`Welcome back, ${user.displayName || "Explorer"}!`);
      }
      onClose();
    } catch (err) {
      console.error("Google login error:", err);
      if (err.code === "auth/popup-closed-by-user") {
        setAuthError("Sign-in cancelled by user.");
      } else if (err.code === "auth/unauthorized-domain") {
        setAuthError("This domain is not yet authorized in Firebase Console.");
      } else {
        setAuthError(err.message || "Failed to sign in with Google.");
      }
    } finally {
      setLoadingProvider(null);
    }
  }

  async function handleGithubLogin() {
    setLoadingProvider("github");
    setAuthError("");
    try {
      const user = await loginWithGithub();
      if (onShowToast) {
        onShowToast(`Welcome back, ${user.displayName || "Explorer"}!`);
      }
      onClose();
    } catch (err) {
      console.error("GitHub login error:", err);
      if (err.code === "auth/popup-closed-by-user") {
        setAuthError("Sign-in cancelled by user.");
      } else if (err.code === "auth/unauthorized-domain") {
        setAuthError("This domain is not yet authorized in Firebase Console.");
      } else if (err.code === "auth/account-exists-with-different-credential") {
        setAuthError("An account already exists with the same email address using Google.");
      } else {
        setAuthError(err.message || "Failed to sign in with GitHub.");
      }
    } finally {
      setLoadingProvider(null);
    }
  }

  async function handleLogout() {
    setLoadingProvider("logout");
    try {
      await logoutUser();
      if (onShowToast) {
        onShowToast("Logged out successfully.");
      }
      onClose();
    } catch (err) {
      console.error("Logout error:", err);
    } finally {
      setLoadingProvider(null);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="login-modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
          ✕
        </button>

        {/* Lottie Vector Animation Header */}
        <div className="login-animation-wrap">
          <div
            ref={animContainerRef}
            style={{ width: "160px", height: "160px", margin: "0 auto" }}
          />
        </div>

        <div className="login-modal-body">
          {currentUser ? (
            /* Logged In User State */
            <div className="user-logged-in-panel">
              <div className="prototype-badge" style={{ background: "rgba(0, 200, 83, 0.08)", borderColor: "rgba(0, 200, 83, 0.25)", color: "#00A844" }}>
                <span className="pulse-dot" style={{ background: "#00C853" }}></span>
                <span>AUTHENTICATED RESEARCHER &bull; ONLINE</span>
              </div>

              <div className="user-profile-card">
                <div className="user-profile-avatar">
                  {currentUser.photoURL ? (
                    <img src={currentUser.photoURL} alt="Avatar" className="user-profile-img" />
                  ) : (
                    <div className="user-profile-initial">
                      {(currentUser.displayName || currentUser.email || "U")[0].toUpperCase()}
                    </div>
                  )}
                </div>
                <div className="user-profile-details">
                  <h3 className="user-profile-name">{currentUser.displayName || "SatQuery Explorer"}</h3>
                  <p className="user-profile-email">{currentUser.email}</p>
                  <div className="user-profile-meta">
                    <span className="meta-tag provider-tag">
                      {currentUser.providerData?.[0]?.providerId === "github.com" ? "GitHub SSO" : "Google Account"}
                    </span>
                    <span className="meta-tag db-tag">Synced to Firestore</span>
                  </div>
                </div>
              </div>

              <div className="login-modal-actions" style={{ marginTop: "16px", width: "100%" }}>
                <button
                  type="button"
                  className="auth-provider-btn logout-btn"
                  onClick={handleLogout}
                  disabled={loadingProvider === "logout"}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                    <polyline points="16 17 21 12 16 7"></polyline>
                    <line x1="21" y1="12" x2="9" y2="12"></line>
                  </svg>
                  <span>{loadingProvider === "logout" ? "Signing Out..." : "Sign Out of SatQuery AI"}</span>
                </button>
              </div>
            </div>
          ) : (
            /* Logged Out / Sign In Options */
            <>
              <div className="prototype-badge">
                <span className="pulse-dot"></span>
                <span>SECURE ACCESS &bull; FIREBASE AUTH &amp; FIRESTORE</span>
              </div>

              <h2 className="login-modal-title">Sign In to SatQuery AI</h2>

              <p className="login-modal-desc">
                Connect your account to save multi-sensor queries, log geospatial detections, and sync workspace data.
              </p>

              {authError && (
                <div className="auth-error-banner">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>{authError}</span>
                </div>
              )}

              {/* SSO Buttons */}
              <div className="auth-btn-stack">
                <button
                  type="button"
                  className="auth-provider-btn google-auth-btn"
                  onClick={handleGoogleLogin}
                  disabled={!!loadingProvider}
                >
                  {loadingProvider === "google" ? (
                    <span className="auth-spinner"></span>
                  ) : (
                    <svg className="provider-logo" viewBox="0 0 24 24">
                      <path fill="#4285F4" d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.665-5.17 3.665-9.17z"/>
                      <path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.33 24 12 24z"/>
                      <path fill="#FBBC05" d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.18 0 9.98 0 12s.45 3.82 1.25 5.42l4.03-3.15z"/>
                      <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.33 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"/>
                    </svg>
                  )}
                  <span>{loadingProvider === "google" ? "Authenticating Google..." : "Continue with Google"}</span>
                </button>

                <button
                  type="button"
                  className="auth-provider-btn github-auth-btn"
                  onClick={handleGithubLogin}
                  disabled={!!loadingProvider}
                >
                  {loadingProvider === "github" ? (
                    <span className="auth-spinner"></span>
                  ) : (
                    <svg className="provider-logo" viewBox="0 0 24 24" fill="currentColor">
                      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
                    </svg>
                  )}
                  <span>{loadingProvider === "github" ? "Authenticating GitHub..." : "Continue with GitHub"}</span>
                </button>
              </div>
            </>
          )}

          <div className="prototype-info-box" style={{ marginTop: "16px" }}>
            <div className="info-icon">⚡</div>
            <div className="info-text">
              <strong>Database Cloud Sync:</strong> All authenticated user sessions are encrypted and securely synchronized to <strong>Google Cloud Firestore</strong>.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
