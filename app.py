import io
import json
import os

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

MODEL_NAME = "gemini-1.5-flash"


def get_api_key():
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None

    return secret_key or os.getenv("GEMINI_API_KEY")


def extract_resume_text(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type or ""
    file_name = uploaded_file.name.lower()

    if file_type == "application/pdf" or file_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)

    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file_name.endswith(".docx")
    ):
        document = Document(io.BytesIO(file_bytes))
        parts = [p.text for p in document.paragraphs if p.text.strip()]

        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))

        text = "\n".join(parts)

    elif file_name.endswith((".txt", ".md")) or file_type.startswith("text/"):
        text = file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")

    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()

    if not text:
        raise ValueError(
            "No readable text was found. If your PDF is a scanned image, "
            "please use a text-based PDF or DOCX."
        )

    return text


def analyze_resume(resume_text, job_description, api_key):
    """Ask Gemini for a structured ATS-style analysis."""
    client = genai.Client(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "ats_score": {
                "type": "integer",
                "description": "Overall ATS readiness score from 0 to 100."
            },
            "summary": {
                "type": "string",
                "description": "Short overall assessment of the resume."
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Strong parts of the resume."
            },
            "improvements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["High", "Medium", "Low"]
                        }
                    },
                    "required": ["issue", "recommendation", "priority"]
                },
                "description": "Specific improvements the candidate should make."
            },
            "keywords_missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important job-related keywords missing or weakly represented."
            },
            "ats_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "check": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["Good", "Needs improvement", "Risk"]
                        },
                        "explanation": {"type": "string"}
                    },
                    "required": ["check", "status", "explanation"]
                },
                "description": "Checks for formatting, sections, keywords, readability, and measurable achievements."
            }
        },
        "required": [
            "ats_score",
            "summary",
            "strengths",
            "improvements",
            "keywords_missing",
            "ats_checks"
        ]
    }

    job_context = (
        job_description.strip()
        if job_description and job_description.strip()
        else "No specific job description was provided. Evaluate general ATS readiness."
    )

    prompt = f"""
You are an expert resume and ATS analyst.

Analyze the resume below. Produce an ATS-style score from 0 to 100 and
specific, actionable improvements.

Important:
- This is an ATS-readiness estimate, not a score produced by a real ATS vendor.
- Do not invent qualifications, jobs, degrees, dates, or achievements.
- Base every recommendation on the supplied resume and job description.
- Check whether sections are clear and conventional.
- Check keyword relevance, measurable achievements, action verbs, clarity,
  consistency, readability, and possible ATS parsing problems.
- If a job description is supplied, compare the resume against it and identify
  important missing or weak keywords.
- Keep recommendations practical and concise.

JOB DESCRIPTION:
{job_context}

RESUME:
{resume_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    result = json.loads(response.text)
    result["ats_score"] = max(0, min(100, int(result["ats_score"])))
    return result


def show_results(result):
    score = result["ats_score"]

    st.subheader("ATS Score")
    st.progress(score / 100)
    st.metric("Estimated ATS readiness", f"{score}/100")

    st.info(result["summary"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What is already strong")
        for item in result["strengths"]:
            st.markdown(f"• {item}")

    with col2:
        st.subheader("Missing / weak keywords")
        if result["keywords_missing"]:
            for item in result["keywords_missing"]:
                st.markdown(f"• `{item}`")
        else:
            st.write("No major missing keywords were identified.")

    st.subheader("Recommended improvements")
    for index, item in enumerate(result["improvements"], start=1):
        with st.expander(
            f"{index}. {item['issue']}  |  Priority: {item['priority']}"
        ):
            st.write(item["recommendation"])

    st.subheader("ATS checks")
    for check in result["ats_checks"]:
        status = check["status"]
        if status == "Good":
            icon = "✅"
        elif status == "Needs improvement":
            icon = "⚠️"
        else:
            icon = "❌"

        st.markdown(f"**{icon} {check['check']}**")
        st.write(check["explanation"])


st.title("📄 Resume ATS Analyzer")
st.write(
    "Upload your resume to get an estimated ATS score and practical improvements."
)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Gemini model: `{MODEL_NAME}`")
    st.caption(
        "Your resume is processed in memory by this app and sent to Gemini "
        "for analysis. It is not saved by the app."
    )

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, and TXT.",
)

job_description = st.text_area(
    "Optional: paste the job description",
    height=180,
    placeholder="Adding the target job description makes the keyword/ATS analysis more useful.",
)

analyze_button = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True,
)

if analyze_button:
    if uploaded_file is None:
        st.warning("Please upload a resume first.")
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "Gemini API key is missing. Add GEMINI_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )
        st.stop()

    try:
        with st.spinner("Reading the resume and generating ATS analysis..."):
            resume_text = extract_resume_text(uploaded_file)

            # Avoid sending an accidentally huge document to the model.
            if len(resume_text) > 50000:
                resume_text = resume_text[:50000]
                st.warning(
                    "The resume text was very long, so the analysis used the first 50,000 characters."
                )

            result = analyze_resume(resume_text, job_description, api_key)

        st.success("Analysis complete.")
        show_results(result)

    except Exception as exc:
        st.error("The analysis could not be completed.")
        st.exception(exc)
