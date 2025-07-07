Project Report: Hybrid Emotion-Aware Mental Wellness Chatbot
Author: Harika Vemula
Date: June 23, 2025
________________________________________
1. Project Overview
This project involves building and deploying two emotion-aware mental wellness chatbot systems:
A Python + Streamlit–based application that runs locally and deploys on Streamlit Cloud via GitHub.
The goal is to create scalable, accessible, and empathetic AI-driven interfaces for emotional support, journaling, and self-reflection.
________________________________________
2. Streamlit + Python–Based Chatbot
This system offers a fully self-contained mental health journaling platform with built-in emotion analysis and visualizations.
Users log in, write entries, receive feedback, and track their mood over time — all through a clean, Streamlit-powered interface.
🧰 Key Components:
•	Custom login/signup using session_state
•	Emotion detection (TF-IDF + Logistic Regression)
•	Chatbot-style responses mapped to emotion categories
•	Journaling + analytics with pie charts and trends
•	Cloud deployment using GitHub + Streamlit Cloud
Login page:
![image](https://github.com/user-attachments/assets/c76d4392-5ba6-430b-a03f-6a8a5e1929a0)
![image](https://github.com/user-attachments/assets/3bdececa-5284-496d-8130-f191b2ca5082)
![image](https://github.com/user-attachments/assets/3bdececa-5284-496d-8130-f191b2ca5082)

After login:
 ![image](https://github.com/user-attachments/assets/0df4f796-4a24-41bd-a37a-dd09a03b0823)
![image](https://github.com/user-attachments/assets/0df4f796-4a24-41bd-a37a-dd09a03b0823)

 ![image](https://github.com/user-attachments/assets/d46dc837-9047-4e48-add2-a66721f16384)
![image](https://github.com/user-attachments/assets/d46dc837-9047-4e48-add2-a66721f16384)
 
Templates: 
    "sadness": "I'm really sorry you're feeling sad. You're not alone. Would you like to talk more?",
    "joy": "That's wonderful to hear! Celebrate these good moments!",
    "anger": "It's okay to feel angry sometimes. Let's try some deep breaths.",
    "fear": "Feeling scared is natural. You're safe here — let's work through this together.",
    "surprise": "That must have been unexpected! How do you feel about it now?",
    "love": "Love is such a powerful emotion. Cherish it and let it guide your actions."
________________________________________
4. How the Chatbot Works
•	User Input (chat or journal) is vectorized using TF-IDF.
•	The trained model predicts an emotion: joy, sadness, anger, etc.
•	Based on the emotion:
o	Dialogflow → triggers a webhook → returns a dynamic reply
o	Streamlit → displays an in-app response + saves the entry
•	Emotion logs are used for charting mood history
________________________________________
5. System Architecture
🖥 Streamlit Flow:
User Login → Input Text → Emotion Detection → Response Generator
             ↓
          Log & Visualize → Mood Analytics
6. Segment-Wise Takeaways
•	Emotion Detection: Lightweight models like Logistic Regression with TF-IDF perform well for basic classification.
•	Streamlit: Ideal for visual feedback, journaling, and dashboard interactivity.
•	User Experience: Emotionally tuned responses increase relatability and support.
•	Deployment: GitHub + Streamlit Cloud enables easy access without complex setup.
________________________________________
7. Future Enhancements
•	Multilingual emotion detection
•	Store logs in secure Firebase/SQLite backends
•	Use GPT-style LLMs for more nuanced conversation
•	Add voice-based journaling (Speech-to-Text)
•	Integrate with mental health resources and helplines dynamically
________________________________________
8. Conclusion
This hybrid chatbot initiative demonstrates that interpretable, lightweight AI can effectively support emotional wellness. By combining Dialogflow's conversational NLP with Streamlit’s interactivity, we created scalable tools for clinics, NGOs, and individuals looking for digital well-being support. 
________________________________________

