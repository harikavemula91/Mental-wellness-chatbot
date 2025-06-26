import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

# --- User Auth System ---
if "users" not in st.session_state:
    st.session_state["users"] = {
        "harika": "12345",
        "admin": "admin123"
    }

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- Page Selection ---
if not st.session_state["authenticated"]:
    auth_mode = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])

    if auth_mode == "Login":
        st.title("🔐 Login to Mental Wellness Chatbot")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = st.session_state["users"]
            if username in users and users[username] == password:
                st.session_state["authenticated"] = True
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid credentials")

    elif auth_mode == "Sign Up":
        st.title("📝 Create Your Account")
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        if st.button("Sign Up"):
            if new_username in st.session_state["users"]:
                st.warning("Username already exists. Try a different one.")
            else:
                st.session_state["users"][new_username] = new_password
                st.success("Account created! Please go to the Login page.")

# --- Emotion Templates ---
emotion_templates = {
    "sadness": "I'm really sorry you're feeling sad. You're not alone. Would you like to talk more?",
    "joy": "That's wonderful to hear! Celebrate these good moments!",
    "anger": "It's okay to feel angry sometimes. Let's try some deep breaths.",
    "fear": "Feeling scared is natural. You're safe here — let's work through this together.",
    "surprise": "That must have been unexpected! How do you feel about it now?",
    "love": "Love is such a powerful emotion. Cherish it and let it guide your actions."
}

def generate_template_response(emotion):
    return emotion_templates.get(emotion.lower(), "I'm here for you. Please tell me more.")

# --- Load Model ---
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))
emotions_emoji_dict = {
    "anger": "😠", "fear": "😨😱", "joy": "😂",
    "sadness": "😔", "surprise": "😮", "love": "🤗"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# --- Chatbot UI ---
if st.session_state["authenticated"]:
    st.title("🧠 PunniBot - Mental Wellness Assistant")
    st.subheader("How are you feeling today?")

    with st.form(key='emotion_form'):
        user_input = st.text_area("Share your thoughts")
        submit_text = st.form_submit_button(label='Analyze Emotion')

    if submit_text and user_input:
        col1, col2 = st.columns(2)
        prediction = predict_emotions(user_input)
        probability = get_prediction_proba(user_input)

        with col1:
            st.success("Original Text")
            st.write(user_input)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

            # Show emotion-based response
            st.subheader("🧠 Supportive Response")
            st.write(generate_template_response(prediction))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)
else:
    st.info("Please log in to access the chatbot.")
