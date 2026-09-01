import { initializeApp, getApps, getApp } from "firebase/app";
import {
  getAuth,
  GoogleAuthProvider,
  GithubAuthProvider,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
} from "firebase/auth";
import {
  getFirestore,
  doc,
  setDoc,
  getDoc,
  serverTimestamp,
} from "firebase/firestore";
import { getAnalytics, isSupported } from "firebase/analytics";

// Read Firebase Config strictly from environment variables (Zero hardcoded secrets)
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
};

// Guard: skip Firebase init entirely if env vars are not configured.
// This lets the app run in pure ML-demo mode without a Firebase project.
const firebaseConfigured = !!(firebaseConfig.apiKey && firebaseConfig.projectId && firebaseConfig.appId);

let app, auth, db;
if (firebaseConfigured) {
  app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
  auth = getAuth(app);
  db = getFirestore(app);
} else {
  console.info("[SatQuery] Firebase env vars not set — running in offline/demo mode. Auth & Firestore disabled.");
  app = null;
  auth = null;
  db = null;
}


// Analytics initialization (safe browser environment check)
let analytics = null;
if (typeof window !== "undefined") {
  isSupported()
    .then((supported) => {
      if (supported && firebaseConfig.measurementId) {
        analytics = getAnalytics(app);
      }
    })
    .catch(() => {});
}

// Authentication Providers
const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: "select_account" });

const githubProvider = new GithubAuthProvider();
githubProvider.addScope("read:user");
githubProvider.addScope("user:email");

/**
 * Save or update user profile in Firestore `users` collection
 * @param {import("firebase/auth").User} user
 * @param {string} providerName
 */
export async function syncUserToFirestore(user, providerName = "google") {
  if (!user || !user.uid || !db) return null;
  try {
    const userRef = doc(db, "users", user.uid);
    const userSnap = await getDoc(userRef);

    const userData = {
      uid: user.uid,
      displayName: user.displayName || user.email?.split("@")[0] || "SatQuery Explorer",
      email: user.email || "No public email",
      photoURL: user.photoURL || "",
      provider: providerName,
      lastLoginAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    };

    if (!userSnap.exists()) {
      userData.createdAt = serverTimestamp();
      userData.role = "researcher";
      userData.queriesCount = 0;
    }

    await setDoc(userRef, userData, { merge: true });
    return userData;
  } catch (error) {
    console.warn("Firestore user sync warning:", error);
    return null;
  }
}

/**
 * Sign in with Google Popup
 */
export async function loginWithGoogle() {
  if (!auth) { console.warn("[SatQuery] Firebase not configured. Add VITE_FIREBASE_* to frontend/.env.local"); return null; }
  const result = await signInWithPopup(auth, googleProvider);
  const user = result.user;
  await syncUserToFirestore(user, "google");
  return user;
}

/**
 * Sign in with GitHub Popup
 */
export async function loginWithGithub() {
  if (!auth) { console.warn("[SatQuery] Firebase not configured. Add VITE_FIREBASE_* to frontend/.env.local"); return null; }
  const result = await signInWithPopup(auth, githubProvider);
  const user = result.user;
  await syncUserToFirestore(user, "github");
  return user;
}

/**
 * Sign out current user
 */
export async function logoutUser() {
  if (!auth) return;
  return signOut(auth);
}

export { app, auth, db, analytics, onAuthStateChanged };
