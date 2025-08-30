import streamlit as st
import hashlib, json, os
from typing import Dict, Optional

USERS_FILE = "users.json"

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _load_users() -> Dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    defaults = {
        "admin": {"password_hash": _hash_password("admin123"), "name": "Administrator", "role": "admin"},
        "user": {"password_hash": _hash_password("user123"), "name": "Demo User", "role": "user"},
    }
    _save_users(defaults)
    return defaults

def _save_users(data: Dict):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

class UserAuth:
    def __init__(self):
        # load users
        self.users = _load_users()

    def authenticate(self, u, p) -> Optional[Dict]:
        if u in self.users and self.users[u]["password_hash"] == _hash_password(p):
            return {"username": u, "name": self.users[u].get("name", u), "role": self.users[u].get("role", "user")}
        return None

    def register_user(self, u, p, n, role="user") -> bool:
        if u in self.users:
            return False
        self.users[u] = {"password_hash": _hash_password(p), "name": n, "role": role}
        _save_users(self.users)
        return True

auth_mgr = UserAuth()

# ------------------- UI -------------------
def show_login_page():
    st.markdown(
        """
        <style>
            body { background-color: #f9fafb; font-family: "Segoe UI", sans-serif; }
            .topbar {
                display: flex; justify-content: space-between; align-items: center;
                padding: 1rem 2rem; background: #ffffff;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }
            .brand { text-align: center; font-size: 1.6rem; font-weight:700; color:#2563eb; }
            .hero { text-align: center; margin-top: 6rem; }
            .hero h1 { font-size: 2.6rem; font-weight:700; color:#1e293b; }   /* dark slate */
            .hero p { font-size: 1.15rem; color:#64748b; margin-top:0.5rem; } /* softer gray-blue */
            .card {
                max-width: 420px; margin: 2rem auto; padding: 1.5rem;
                background: #ffffff; border-radius: 12px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            }
            .stButton button {
                background: #2563eb !important;
                color: white !important;
                border-radius: 8px !important;
                padding: 0.4rem 0.9rem;
                font-weight: 500;
            }
            .stButton button:hover {
                background: #1d4ed8 !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------- TOP BAR ----------
    col1, col2, col3 = st.columns([2,4,2])
    with col2:
        st.markdown('<div class="brand">📘 StudyMate</div>', unsafe_allow_html=True)
    with col3:
        if "show_auth" not in st.session_state:
            st.session_state.show_auth = None
        if st.button("Login"):
            st.session_state.show_auth = "login"; st.rerun()
        if st.button("Register"):
            st.session_state.show_auth = "signup"; st.rerun()

    # ---------- HERO ----------
    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to StudyMate</h1>
            <p>Your AI-powered study assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- LOGIN / SIGNUP ----------
    if st.session_state.show_auth:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.session_state.show_auth == "login":
            st.subheader("🔑 Login")
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In"):
                    info = auth_mgr.authenticate(u, p)
                    if info:
                        st.session_state.logged_in = True
                        st.session_state.user_info = info
                        st.session_state.show_auth = None
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        elif st.session_state.show_auth == "signup":
            st.subheader("📝 Register")
            with st.form("register_form"):
                nu = st.text_input("New Username")
                np = st.text_input("New Password", type="password")
                nn = st.text_input("Full Name")
                if st.form_submit_button("Create Account"):
                    if nu and np and nn:
                        if auth_mgr.register_user(nu, np, nn):
                            st.success("✅ Account created! Please login.")
                            st.session_state.show_auth = "login"
                        else:
                            st.error("⚠ Username already exists.")
                    else:
                        st.error("Please fill all fields.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.info("Demo Login → *admin/admin123* | *user/user123*")


def check_login() -> bool:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        show_login_page()
        return False
    return True


def logout():
    st.session_state.logged_in = False
    st.session_state.pop("user_info", None)
    st.session_state.show_auth = None
    st.rerun()
