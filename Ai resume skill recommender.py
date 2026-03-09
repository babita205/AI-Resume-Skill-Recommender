# ==============================
# INSTALL LIBRARIES
# ==============================

!pip install gradio scikit-learn pandas PyPDF2

import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import PyPDF2

# ==============================
# SKILL DATABASE
# ==============================

skill_profiles = {

    "Web Developer": [
        "HTML","CSS","JavaScript","React","NodeJS","MongoDB"
    ],

    "Data Scientist": [
        "Python","Pandas","NumPy","Machine Learning","SQL","Data Visualization"
    ],

    "Machine Learning Engineer": [
        "Python","TensorFlow","PyTorch","Deep Learning","Statistics","Machine Learning"
    ],

    "Backend Developer": [
        "Python","Django","Flask","SQL","Docker","APIs"
    ],

    "AI Engineer": [
        "Python","Machine Learning","Deep Learning","NLP","TensorFlow","PyTorch"
    ]

}

# ==============================
# COMMON SKILLS LIST
# ==============================

common_skills = [
"Python","Java","C++","HTML","CSS","JavaScript","React","NodeJS",
"SQL","Machine Learning","Deep Learning","TensorFlow","PyTorch",
"Django","Flask","Pandas","NumPy","Docker","APIs","NLP"
]

# ==============================
# SKILL RECOMMENDER
# ==============================

def recommend_skills(user_skills):

    user_skills = [s.strip() for s in user_skills.split(",")]

    profiles = []
    roles = []

    for role, skills in skill_profiles.items():
        profiles.append(" ".join(skills))
        roles.append(role)

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(profiles)

    user_vector = vectorizer.transform([" ".join(user_skills)])

    similarity = cosine_similarity(user_vector, vectors)

    best_index = similarity.argmax()

    best_role = roles[best_index]

    recommended = skill_profiles[best_role]

    missing = list(set(recommended) - set(user_skills))

    advice = career_advice(best_role)

    result = f"Suggested Career Path: {best_role}\n\nRecommended Skills:\n"

    for skill in missing:
        result += f"- {skill}\n"

    result += f"\nCareer Advice:\n{advice}"

    return result

# ==============================
# RESUME PARSER
# ==============================

def extract_skills_from_resume(file):

    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    found_skills = []

    for skill in common_skills:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    if len(found_skills) == 0:
        return "No common skills detected."

    role, missing = recommend_role(found_skills)

    result = f"Detected Skills:\n{found_skills}\n\nSuggested Career Path: {role}\n\nRecommended Skills:\n"

    for skill in missing:
        result += f"- {skill}\n"

    return result

# ==============================
# ROLE PREDICTION
# ==============================

def recommend_role(user_skills):

    profiles = []
    roles = []

    for role, skills in skill_profiles.items():
        profiles.append(" ".join(skills))
        roles.append(role)

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(profiles)

    user_vector = vectorizer.transform([" ".join(user_skills)])

    similarity = cosine_similarity(user_vector, vectors)

    best_index = similarity.argmax()

    best_role = roles[best_index]

    recommended = skill_profiles[best_role]

    missing = list(set(recommended) - set(user_skills))

    return best_role, missing

# ==============================
# CAREER ADVICE
# ==============================

def career_advice(role):

    tips = {

        "Web Developer":
        "Build full-stack projects and master React + backend frameworks.",

        "Data Scientist":
        "Focus on data analysis, statistics, and machine learning projects.",

        "Machine Learning Engineer":
        "Work on ML pipelines, deep learning models, and deployment.",

        "Backend Developer":
        "Learn APIs, databases, and scalable backend systems.",

        "AI Engineer":
        "Focus on deep learning, NLP, and generative AI systems."
    }

    return tips.get(role,"Keep building projects and learning new technologies.")

# ==============================
# GRADIO UI
# ==============================

skill_ui = gr.Interface(
    fn=recommend_skills,
    inputs=gr.Textbox(label="Enter your skills (comma separated)"),
    outputs="text",
    title="AI Resume Skill Recommender",
    description="Enter your skills and get AI-based career suggestions."
)

resume_ui = gr.Interface(
    fn=extract_skills_from_resume,
    inputs=gr.File(label="Upload Resume (PDF)"),
    outputs="text",
    title="AI Resume Analyzer",
    description="Upload your resume and get skill recommendations."
)

app = gr.TabbedInterface(
    [skill_ui, resume_ui],
    ["Skill Recommender", "Resume Analyzer"]
)

app.launch()
