import streamlit as st
import requests

st.set_page_config(page_title="Russian to English Translator", layout="centered")
st.title("Russian to English Translator")

if "page" not in st.session_state:
    st.session_state.page = "login"
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

def login_page():
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter username", key="login_username")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")
        login_button = st.form_submit_button("Login", use_container_width=True)

        if login_button:
            if not username or not password:
                st.error("Please fill in both username and password")
                return
            try:
                response = requests.post(
                    "http://localhost:8000/token",
                    json={"username": username, "password": password}
                )
                response.raise_for_status()
                token = response.json().get("access_token")
                st.session_state.token = token
                st.session_state.username = username
                st.session_state.page = "translate"
                st.success(f"Welcome, {username}!")
                st.rerun()
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Invalid username or password")
                elif e.response.status_code == 422:
                    st.error("Invalid data format. Please check the input fields.")
                else:
                    st.error(f"Login error: {e}")
            except requests.RequestException as e:
                st.error(f"Connection error: {e}")

    if st.button("Don't have an account? Register", key="to_register"):
        st.session_state.page = "register"
        st.rerun()

def register_page():
    st.subheader("Register")
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter username", key="register_username")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password", key="register_confirm_password")
        register_button = st.form_submit_button("Register", use_container_width=True)

        if register_button:
            if not username or not password or not confirm_password:
                st.error("Please fill in all fields")
                return
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            try:
                response = requests.post(
                    "http://localhost:8000/register",
                    json={"username": username, "password": password}
                )
                response.raise_for_status()
                st.success("Registration successful! Please log in.")
                st.session_state.page = "login"
                st.rerun()
            except requests.HTTPError as e:
                if e.response.status_code == 400:
                    st.error("User already exists")
                elif e.response.status_code == 422:
                    st.error("Invalid data format. Please check the input fields.")
                else:
                    st.error(f"Registration error: {e}")
            except requests.RequestException as e:
                st.error(f"Connection error: {e}")

    if st.button("Already have an account? Login", key="to_login"):
        st.session_state.page = "login"
        st.rerun()

def translate_page():
    if not st.session_state.token:
        st.error("Please log in to continue")
        st.session_state.page = "login"
        st.rerun()
        return

    st.subheader(f"Translation (user: {st.session_state.username})")
    # Выбор типа ввода вне формы
    input_type = st.radio("Select input type", ("Text", "File"), key="translate_input_type")

    with st.form("translate_form", clear_on_submit=True):
        input_text = None
        file = None
        if input_type == "Text":
            input_text = st.text_area(
                "Russian Text",
                placeholder="Enter text in Russian (max 1000 characters)...",
                height=150,
                max_chars=1000,
                key="translate_text"
            )
        else:
            file = st.file_uploader(
                "Upload a .txt file (max 1000 characters)",
                type=["txt"],
                key="translate_file"
            )

        submit_button = st.form_submit_button("Translate", use_container_width=True)

        if submit_button:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                if input_type == "Text":
                    if not input_text or not input_text.strip():
                        st.error("Text must not be empty")
                        return
                    if len(input_text.strip()) > 1000:
                        st.error("Text exceeds 1000 characters")
                        return
                    response = requests.post(
                        "http://localhost:8000/translate",
                        json={"text": input_text.strip()},
                        headers={**headers, "Content-Type": "application/json"}
                    )
                else:
                    if not file:
                        st.error("No file uploaded")
                        return
                    file_content = file.read().decode("utf-8")
                    if not file_content.strip():
                        st.error("File must not be empty")
                        return
                    if len(file_content.strip()) > 1000:
                        st.error("File content exceeds 1000 characters")
                        return
                    response = requests.post(
                        "http://localhost:8000/translate",
                        files={"file": (file.name, file_content.encode("utf-8"), "text/plain")},
                        headers=headers
                    )

                response.raise_for_status()
                result = response.json()
                translation = result.get("translation", "Translation not received")
                if input_type == "File" and result.get("s3_path"):
                    st.success(f"File uploaded to S3: {result['s3_path']}")
                st.text_area("English Translation", value=translation, height=150, disabled=True)
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Session expired. Please log in again.")
                    st.session_state.token = None
                    st.session_state.username = None
                    st.session_state.page = "login"
                    st.rerun()
                elif e.response.status_code == 422:
                    st.error("Invalid data format. Please check the input.")
                elif e.response.status_code == 400:
                    st.error(f"Error: {e.response.json().get('detail', 'Invalid request')}")
                else:
                    st.error(f"Error: {e}")
            except requests.RequestException as e:
                st.error(f"Connection error: {e}")

    if st.button("Logout", key="logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.page = "login"
        st.success("You have logged out")
        st.rerun()

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "translate":
    translate_page()

st.markdown("---")
st.markdown("Project by German Berezin")