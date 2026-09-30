import streamlit as st
import os
from dotenv import load_dotenv
from google import genai

#====================================
# Load Environment
#====================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Please add google api key")

#====================================
# Gemini client
#===================================

client = genai.Client(
    api_key = api_key
)

st.set_page_config(
    page_title="AI Email Generator",
    page_icon="✉️",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="title">
        ✉️ AI Email Generator
    </div>

    <div class="subtitle">
        Generate professional emails using Gemini
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# EMAIL DETAILS
# =========================================================

col1, col2 = st.columns(2)

with col1:
    recipient = st.text_input("Recipient", placeholder="e.g. John Smith")

with col2:
    email_type = st.selectbox(
        "Email Type",
        [
            "Professional",
            "Follow-up",
            "Meeting Request",
            "Job Application",
            "Sales",
            "Thank You",
            "Apology",
            "Reminder",
            "Leave Request",
            "Custom"
        ]
    )

tone = st.selectbox(
    "Tone",
    [
        "Professional",
        "Friendly",
        "Formal",
        "Polite",
        "Persuasive",
        "Concise"
    ]
)

language = st.selectbox(
    "Language",
    [
        "English",
        "Hindi",
        "Gujarati"

    ]
)

request = st.text_area(
    "Write your email",
    placeholder=(
        "Example: Ask the client for meeting"
    )
)

sender_name = st.text_input(
    "Your Name",
    placeholder="e.g. John"
)

import time


def generate_email(prompt):

    models = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    ]

    last_error = None

    for model in models:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                return response.text

            except Exception as e:

                last_error = e

                error_text = str(e)

                # Retry temporary Gemini availability errors
                if "503" in error_text or "UNAVAILABLE" in error_text:

                    if attempt < 2:

                        time.sleep(2 ** attempt)

                        continue

                    break

                # Don't retry other errors
                raise e

    raise last_error


if st.button(
    "Generate Email",
    type="primary",
    use_container_width=True
):
    if not request.strip():
        st.warning("Please write email")
        st.stop()

    prompt=f""""
    you are a professional email writer
    Generate email based on given information below

    Recipient:
    {recipient}

    Email Type
    {email_type}

    Tone:
    {tone}

    Language:
    {language}

    Sender:
    {sender_name}

    Instructions:

    1. Read request content properly
    2. Write complete and professional email
    4. Use the requested language
    5. Use the requested tone
    7. Use only information provided by the user
    8. Read email request content and make email content as per requested

    Subject:<subject>

    <email body>


    """

    with st.spinner("Generating email..."):
        try:
            email_content = generate_email(prompt)
            st.write(email_content)

        except Exception as e:
            st.error(f"Error generating email: {e}")
