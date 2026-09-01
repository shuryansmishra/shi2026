import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar.jsx";
import SingleImageVQA from "./components/SingleImageVQA.jsx";
import ChangeDetection from "./components/ChangeDetection.jsx";
import LiveMapSelection from "./components/LiveMapSelection.jsx";
import LoginModal from "./components/LoginModal.jsx";
import Toast from "./components/Toast.jsx";
import { checkHealth } from "./api.js";
import { auth, onAuthStateChanged, logoutUser } from "./firebase.js";

// Read Mapbox token strictly from environment variable (.env file)
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || "";

export default function App() {
  const [activeTab, setActiveTab] = useState("change_detection"); // Default to Change Detection as in Figma
  const [toastMessage, setToastMessage] = useState("");
  const [toastVisible, setToastVisible] = useState(false);
  const [backendStatus, setBackendStatus] = useState("ok");
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);

  useEffect(() => {
    // Listen to Firebase Auth state
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
    });

    // Ping backend health
    checkHealth().then((res) => {
      if (res.status === "ok") {
        setBackendStatus("ok");
      } else {
        setBackendStatus("active");
      }
    });

    return () => unsubscribe();
  }, []);

  function showToast(msg) {
    setToastMessage(msg);
    setToastVisible(true);
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => {
      setToastVisible(false);
    }, 2800);
  }

  async function handleLogout() {
    try {
      await logoutUser();
      showToast("Logged out successfully.");
    } catch (err) {
      console.error("Logout failed:", err);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient-backdrop" />

      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        onTabChange={(tab) => {
          setActiveTab(tab);
        }}
        backendStatus={backendStatus}
        onOpenLogin={() => setIsLoginOpen(true)}
        currentUser={currentUser}
        onLogout={handleLogout}
      />

      {/* Dynamic View Sections with Smooth Animated Transitions */}
      <main style={{ position: "relative", zIndex: 1, flex: 1, display: "flex", flexDirection: "column" }}>
        <div key={activeTab} className="tab-view-enter">
          {activeTab === "single_vqa" && (
            <SingleImageVQA
              mapboxToken={MAPBOX_TOKEN}
              onShowToast={showToast}
            />
          )}

          {activeTab === "change_detection" && (
            <ChangeDetection
              mapboxToken={MAPBOX_TOKEN}
              onShowToast={showToast}
            />
          )}

          {activeTab === "live_map" && (
            <LiveMapSelection
              mapboxToken={MAPBOX_TOKEN}
              onShowToast={showToast}
            />
          )}
        </div>
      </main>

      {/* Global Interactive Toast Notification */}
      <Toast message={toastMessage} visible={toastVisible} />

      {/* Login & Researcher Profile Modal */}
      <LoginModal
        isOpen={isLoginOpen}
        onClose={() => setIsLoginOpen(false)}
        currentUser={currentUser}
        onShowToast={showToast}
      />
    </div>
  );
}
