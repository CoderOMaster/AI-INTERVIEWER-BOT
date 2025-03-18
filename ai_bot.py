import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------------------
# Load the LLaMA model and tokenizer
# ---------------------------------------
st.set_page_config(
    page_title="Automated Interview Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """
    Loads the LLaMA model and tokenizer from Hugging Face.
    Adjust the model_name as per your access.
    """
    model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="cpu", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    return model, tokenizer

# Load model and tokenizer
with st.spinner("Loading LLaMA model... This may take a few minutes."):
    model, tokenizer = load_model()

# ---------------------------------------
# Streamlit App Configuration
# ---------------------------------------

# App Title
st.title("🤖 Automated Interview Bot")

# Sidebar Navigation
st.sidebar.title("Navigation")
steps = ["Job Description", "Technical Interview", "HR Interview", "Results"]
selection = st.sidebar.radio("Go to", steps)

# Initialize session state variables
if 'tech_score' not in st.session_state:
    st.session_state['tech_score'] = 0
if 'hr_score' not in st.session_state:
    st.session_state['hr_score'] = 0
if 'technical_questions' not in st.session_state:
    st.session_state['technical_questions'] = []
if 'hr_questions' not in st.session_state:
    st.session_state['hr_questions'] = [
        "Why do you want to work at our company?",
        "Describe a challenging situation you faced and how you handled it.",
        "Where do you see yourself in five years?",
        "How do you handle feedback and criticism?",
        "Can you describe your ideal work environment?"
    ]
if 'hr_answers' not in st.session_state:
    st.session_state['hr_answers'] = {}
if 'technical_answers' not in st.session_state:
    st.session_state['technical_answers'] = {}
if 'job_description' not in st.session_state:
    st.session_state['job_description'] = ""

# ---------------------------------------
# Function Definitions
# ---------------------------------------

def generate_questions(job_description, num_questions=5):
    """
    Generates technical questions based on the job description using the LLaMA model.
    """
    prompt = (
        f"Based on the following job description, generate {num_questions} technical interview questions:\n\n"
        f"Job Description:\n{job_description}\n\nQuestions:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract questions
    if "Questions:" in generated_text:
        questions_part = generated_text.split("Questions:")[1]
    else:
        questions_part = generated_text
    questions = questions_part.strip().split('\n')
    # Clean and format questions
    formatted_questions = []
    for q in questions:
        q = q.strip()
        if q:
            # Remove numbering if present
            q = q.lstrip("1. ").lstrip("2. ").lstrip("3. ").lstrip("4. ").lstrip("5. ")
            formatted_questions.append(q)
        if len(formatted_questions) >= num_questions:
            break
    return formatted_questions[:num_questions]

def evaluate_answer(question, answer):
    """
    Evaluates the candidate's answer using the LLaMA model.
    Returns a tuple (is_correct: bool, feedback: str).
    """
    prompt = (
        f"Question: {question}\n"
        f"Candidate Answer: {answer}\n"
        f"Is the candidate's answer correct and satisfactory? Provide a brief explanation."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.0,  # Deterministic output
            top_p=0.95,
            num_return_sequences=1,
            do_sample=False
        )
    evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Simple heuristic: Check for affirmative words
    affirmative = ["yes", "correct", "satisfactory", "adequate", "good"]
    negative = ["no", "incorrect", "unsatisfactory", "inadequate", "bad"]
    eval_lower = evaluation.lower()
    is_correct = False
    for word in affirmative:
        if word in eval_lower:
            is_correct = True
            break
    return is_correct, evaluation

# ---------------------------------------
# Job Description Page
# ---------------------------------------
if selection == "Job Description":
    st.header("📄 Step 1: Job Description")
    job_description_input = st.text_area(
        "Enter the Job Description:",
        height=300,
        placeholder="Paste or type the job description here..."
    )
    if st.button("Generate Technical Questions"):
        if job_description_input.strip() == "":
            st.error("Please enter a valid job description.")
        else:
            with st.spinner("Generating technical questions..."):
                technical_questions = generate_questions(job_description_input)
            st.session_state['technical_questions'] = technical_questions
            st.session_state['job_description'] = job_description_input
            st.success("Technical questions generated successfully!")
            st.write("### Generated Technical Questions:")
            for idx, question in enumerate(technical_questions, 1):
                st.write(f"**Q{idx}:** {question}")

# ---------------------------------------
# Technical Interview Page
# ---------------------------------------
if selection == "Technical Interview":
    if not st.session_state['technical_questions']:
        st.warning("⚠️ Please generate technical questions first in the 'Job Description' section.")
    else:
        st.header("🖥️ Step 2: Technical Interview")
        with st.form("technical_form"):
            st.write("Please answer the following technical questions:")
            for idx, question in enumerate(st.session_state['technical_questions'], 1):
                answer = st.text_area(f"**Q{idx}:** {question}", key=f"tech_{idx}", height=100)
                st.session_state['technical_answers'][question] = answer
            submitted = st.form_submit_button("Submit Technical Answers")
            if submitted:
                st.write("## Technical Interview Evaluation")
                correct_answers = 0
                for question, answer in st.session_state['technical_answers'].items():
                    if answer.strip() == "":
                        st.write(f"**Question:** {question}")
                        st.write("**Your Answer:** *No answer provided.*")
                        st.write("**Evaluation:** ❌ No answer provided.\n")
                        continue
                    is_correct, feedback = evaluate_answer(question, answer)
                    st.write(f"**Question:** {question}")
                    st.write(f"**Your Answer:** {answer}")
                    if is_correct:
                        st.write(f"**Evaluation:** ✅ Correct")
                        correct_answers += 1
                    else:
                        st.write(f"**Evaluation:** ❌ Incorrect")
                    st.write(f"**Feedback:** {feedback}\n")
                st.session_state['tech_score'] = correct_answers
                if correct_answers >= 4:
                    st.success(f"✅ Technical Interview Passed! Score: {correct_answers}/5")
                else:
                    st.error(f"❌ Technical Interview Failed. Score: {correct_answers}/5")

# ---------------------------------------
# HR Interview Page
# ---------------------------------------
if selection == "HR Interview":
    if st.session_state['tech_score'] < 4:
        st.warning("⚠️ Candidate did not pass the Technical Interview. Cannot proceed to HR Interview.")
    else:
        st.header("💼 Step 3: HR Interview")
        with st.form("hr_form"):
            st.write("Please answer the following HR questions:")
            for idx, question in enumerate(st.session_state['hr_questions'], 1):
                answer = st.text_area(f"**Q{idx}:** {question}", key=f"hr_{idx}", height=100)
                st.session_state['hr_answers'][question] = answer
            submitted = st.form_submit_button("Submit HR Answers")
            if submitted:
                st.write("## HR Interview Evaluation")
                correct_answers = 0
                for question, answer in st.session_state['hr_answers'].items():
                    if answer.strip() == "":
                        st.write(f"**Question:** {question}")
                        st.write("**Your Answer:** *No answer provided.*")
                        st.write("**Evaluation:** ❌ No answer provided.\n")
                        continue
                    is_correct, feedback = evaluate_answer(question, answer)
                    st.write(f"**Question:** {question}")
                    st.write(f"**Your Answer:** {answer}")
                    if is_correct:
                        st.write(f"**Evaluation:** ✅ Correct")
                        correct_answers += 1
                    else:
                        st.write(f"**Evaluation:** ❌ Incorrect")
                    st.write(f"**Feedback:** {feedback}\n")
                st.session_state['hr_score'] = correct_answers
                if correct_answers >= 3:
                    st.success(f"🎉 Candidate Selected! HR Score: {correct_answers}/5")
                else:
                    st.error(f"❌ Candidate Not Selected. HR Score: {correct_answers}/5")

# ---------------------------------------
# Results Page
# ---------------------------------------
if selection == "Results":
    st.header("📊 Step 4: Interview Results")
    if st.session_state['tech_score'] == 0 and st.session_state['hr_score'] == 0:
        st.info("No interview has been conducted yet.")
    else:
        st.subheader("🔧 Technical Interview")
        st.write(f"**Score:** {st.session_state['tech_score']}/5")
        if st.session_state['tech_score'] >= 4:
            st.success("**Status:** Passed")
        else:
            st.error("**Status:** Failed")
        
        if st.session_state['tech_score'] >= 4:
            st.subheader("💼 HR Interview")
            st.write(f"**Score:** {st.session_state['hr_score']}/5")
            if st.session_state['hr_score'] >= 3:
                st.success("**Final Decision:** **Selected**")
            else:
                st.error("**Final Decision:** **Not Selected**")
