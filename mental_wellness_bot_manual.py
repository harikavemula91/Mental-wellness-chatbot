"""# Frontend deployment using Streamlit"""

import streamlit as st

# Simple user store
USER_CREDENTIALS = {
    "harika": "mypassword",
    "admin": "admin123"
}

def login():
    st.title("🔒 Login to Mental Wellness Chatbot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    login()
else:
    st.success("You're logged in!")

username = st.secrets["auth"]["username1"]

#removing pip install due to error in in streamlit
import streamlit as st

import altair as alt

pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "fear": "😨😱", "joy": "😂", "sadness": "😔", "surprise": "😮", "love": "🤗"}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Emotion Detection")
    st.subheader("Detect Emotions In user's Statement")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()

