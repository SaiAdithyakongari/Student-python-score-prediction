import streamlit as st
import pandas as pd
import joblib
import altair as alt
import time

st.set_page_config(page_title="Student ML Predictor", layout="wide", page_icon="🎓")

# Custom CSS for styling
st.markdown("""
<style>
    .big-font {
        font-size: 3rem !important;
        font-weight: 700;
        background: -webkit-linear-gradient(#00f0ff, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #1E212A;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #00f0ff;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('bi.csv', encoding='latin1')

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error("Failed to load model or data. Please run `train.py` first.")
    st.stop()

st.markdown('<p class="big-font">Student Performance Analytics & AI</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("🎯 Student Profile")
st.sidebar.write("Adjust the parameters to predict the Python Score.")

age = st.sidebar.slider("Age", min_value=18, max_value=80, value=25)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
country = st.sidebar.selectbox("Country", ["norway", "south africa", "kenya", "uganda"])
residence = st.sidebar.selectbox("Residence", ["private", "sognsvann", "bi residence"])
entry_exam = st.sidebar.slider("Entry Exam Score", 0, 100, 75)
prev_edu = st.sidebar.selectbox("Previous Education", ["high school", "diploma", "bachelors", "masters"])
study_hours = st.sidebar.slider("Study Hours", 0, 300, 150)
db_score = st.sidebar.slider("DB Score", 0, 100, 70)

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [age], 'gender': [gender], 'country': [country],
    'residence': [residence], 'entryEXAM': [entry_exam],
    'prevEducation': [prev_edu], 'studyHOURS': [study_hours], 'DB': [db_score]
})

# Predict
prediction = model.predict(input_data)[0]

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Predicted Python Score", value=f"{prediction:.1f}/100", delta=f"{prediction - df['Python'].mean():.1f} vs Avg")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Study Hours", value=f"{study_hours} hrs")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Database Score", value=f"{db_score}/100")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.subheader("📊 Dataset Context")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Scatter Plot: Study Hours vs DB
    scatter = alt.Chart(df.dropna(subset=['Python', 'DB'])).mark_circle(size=60).encode(
        x='DB:Q',
        y='Python:Q',
        color=alt.Color('gender:N', scale=alt.Scale(scheme='set1')),
        tooltip=['Age', 'studyHOURS', 'DB', 'Python']
    ).interactive().properties(title="DB Score vs Python Score in Cohort")
    st.altair_chart(scatter, use_container_width=True)

with chart_col2:
    # Bar Chart: Average Python Score by Education
    df_clean = df.copy()
    df_clean['prevEducation'] = df_clean['prevEducation'].astype(str).str.lower().str.strip().replace({'highschool': 'high school', 'barrrchelors': 'bachelors'})
    bar = alt.Chart(df_clean.dropna(subset=['Python'])).mark_bar(color='#8a2be2').encode(
        x='prevEducation:N',
        y='mean(Python):Q',
        tooltip=['mean(Python)']
    ).properties(title="Average Python Score by Previous Education")
    st.altair_chart(bar, use_container_width=True)

# --- AI CHATBOT ---
st.write("---")
st.subheader("🤖 AI Academic Advisor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting based on the prediction
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"Hello! I am your AI Academic Advisor. Based on your profile, your predicted Python score is **{prediction:.1f}**. How can I help you improve?"
    })

# Always update the first message if prediction changes significantly, 
# but for simplicity we keep the chat history as is.

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask for advice on how to improve your score..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simulated AI Response Generation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simple rule-based AI logic
        prompt_lower = prompt.lower()
        if "improve" in prompt_lower or "better" in prompt_lower:
            ai_response = f"To improve from your predicted {prediction:.1f}, consider increasing your study hours from {study_hours} to at least 160. Additionally, working on your Database fundamentals (currently at {db_score}) directly correlates with better Python performance!"
        elif "db" in prompt_lower or "database" in prompt_lower:
            ai_response = f"Your DB score is {db_score}. Database concepts often intertwine with backend Python programming. Try doing some SQL + Python practice exercises!"
        elif "hours" in prompt_lower or "study" in prompt_lower:
            ai_response = f"You are currently studying {study_hours} hours. The top performing students usually put in around 155+ hours. Try adding an extra hour to your daily routine!"
        else:
            ai_response = f"That's a great question! As an AI analyzing your profile, I see your entry exam was {entry_exam}. Leverage the same focus you used to achieve that score towards mastering Python data structures."

        # Simulate stream of response with milliseconds delay
        for chunk in ai_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
