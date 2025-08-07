from langchain_community.document_loaders import PDFPlumberLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import streamlit as st
from dotenv import load_dotenv
from streamlit_quill import st_quill
import os

load_dotenv()
model=ChatOpenAI(model="gpt-4o-mini")
model_turbo=ChatOpenAI(model="gpt-3.5-turbo")
st.markdown("<h1 style='text-align: center;'>AI-Powered Resume Screener</h1>", unsafe_allow_html=True)

st.markdown("### 🧑‍💼 Recruiter Information")

recruiter_name = st.text_input("🔤 Enter your name", placeholder="e.g., Abdul Rehman")
recruiter_email = st.text_input("📧 Enter your email", placeholder="e.g., abdul@company.com")
company_name = st.text_input("🏢 Company name", placeholder="e.g., TalentHive")


st.markdown("### 📝 Paste the Job Description")
st.markdown(
    "Please provide the job description for the role you're hiring. "
    "The AI will use this to evaluate how well each candidate fits.")
job_description = st_quill(placeholder="e.g., We're hiring a frontend developer with 2+ years of React experience, HTML/CSS skills, and strong collaboration abilities.")
RESUME_DIR = "1. WorkFlows/3. ConditionalWorkflows/resume"
os.makedirs(RESUME_DIR, exist_ok=True)
resume_file=st.file_uploader(accept_multiple_files=False,label="Enter Applicant's Resume Here")

if resume_file is not None:
    path_resume_file=os.path.join(RESUME_DIR,resume_file.name)
    with open(path_resume_file,"wb") as f:
        f.write(resume_file.getbuffer())

    loader=PDFPlumberLoader(path_resume_file)
    resume=loader.load()
    resume_text="\n".join(page.page_content for page in resume)
    os.remove(path_resume_file)

class ResumeStateSchema(TypedDict):
    job_description:str
    resume:str

    name:str
    email:str

    tone:Literal["confident", "humble", "desperate", "arrogant"]
    fit:Literal["high", "medium", "low"]
    flag:Literal["none", "minor", "major"]

    tone_summary:str
    fit_summary:str
    flag_summary:str

    recruiter_name:str
    recruiter_email:str
    company_name:str

    response:str
class InfoSchema(BaseModel):
    name:Annotated[str,Field(description="Name of the applicant.")]
    email:Annotated[str,Field(description="Email of the applicant.")]
class AnalyzeToneSchema(BaseModel):
    tone:Annotated[Literal["confident", "humble", "desperate", "arrogant"],Field(description="This contains the tone of the applicant.")]
    tone_summary:Annotated[str,Field(description="This is the summary of tone of the person.")]
class AnalyzeFitSchema(BaseModel):
    fit:Annotated[Literal["high", "medium", "low"],Field(description="Based on Job description this define that the applicant fits to this job or not.")]
    fit_summary:Annotated[str,Field(description="This is the summary of how does the applicant fits with job decription.")]
class AnalyzeFlagSchema(BaseModel):
    flag:Annotated[Literal["none", "minor", "major"],Field(description="Based on appliicants resume this define the red flag in applicant.")]
    flag_summary:Annotated[str,Field(description="This is the summary of flags in the applicant.")]

InfoModel=model.with_structured_output(InfoSchema)
ToneModel=model.with_structured_output(AnalyzeToneSchema)
FitModel=model.with_structured_output(AnalyzeFitSchema)
FlagModel=model.with_structured_output(AnalyzeFlagSchema)

def get_applicant_info(state:ResumeStateSchema)->ResumeStateSchema:
    resume=state["resume"]
    prompt=PromptTemplate(
        template="""
    You are provided with the text of aplicant's resume. You have to get his name and email form the text.
    The text of resume is given below:
    {resume_text}
""", input_variables=["resume_text"])
    chain=prompt|InfoModel
    response=chain.invoke({"resume_text":resume})
    return {
        'name':response.name,
        'email':response.email,
        'recruiter_name':recruiter_name,
        'recruiter_email':recruiter_email,
        'company_name':company_name
    }



def analyze_tone(state:ResumeStateSchema)->ResumeStateSchema:
    resume=state["resume"]
    prompt=PromptTemplate(
        template="""
    You will be provided with the text of a resume or a cover letter. You job is to identify the tone of applicant's writting.
    It can be one of them.
    - Confident: clear, assertive, optimistic
    - Humble: polite, modest, respectful
    - Desperate: overly eager, begging tone, emotional
    - Arrogant: overconfident, dismissive, bragging
    Also you have to provide a summary of reason why you selected this label.
    The text of resume is given below:
    {resume_text}
""", input_variables=["resume_text"])
    chain=prompt|ToneModel
    response=chain.invoke({"resume_text":resume})
    return {
        'tone':response.tone,
        'tone_summary':response.tone_summary
    }

def analyze_fit(state:ResumeStateSchema)->ResumeStateSchema:
    job_desc=state["job_description"]
    resume=state["resume"]
    prompt=PromptTemplate(
        template="""
    You will be provided with the job description and text of a resume or a cover letter. You job is to identify that is the applicant fit fot this job.
    It can be one of them.
    - High: Most fit for the job
    - Medium: Lacks in some skills etc
    - Low: Totally different skills
    Also you have to provide a summary of reason why you selected this label.
    The text of resume and Job Description is given below:
    Job Description: {job_description}
    Resume Text: {resume_text}
""", input_variables=["resume_text","job_description"])
    chain=prompt|FitModel
    response=chain.invoke({"resume_text":resume,"job_description":job_desc})
    return {
        'fit':response.fit,
        'fit_summary':response.fit_summary
    }

def analyze_flag(state:ResumeStateSchema)->ResumeStateSchema:
    resume=state["resume"]
    prompt=PromptTemplate(
        template = """
    You will be given the full text of a resume or cover letter.
    Your task is to flag any red signals that may be a concern for recruiters.
    Only choose one of the following labels:
    - Clean: No red flags found
    - Minor Flags: Some small issues (e.g. vague wording, unexplained roles)
    - Major Flags: Strong concerns (e.g. employment gaps, unrealistic claims, job-hopping)
    Also you have to provide a summary of reason why you selected this label.
    Text:
    {resume_text}
    """, input_variables=["resume_text"])
    chain=prompt|FlagModel
    response=chain.invoke({"resume_text":resume})
    return {
        'flag':response.flag,
        "flag_summary":response.flag_summary
    }
    
def route_node(state:ResumeStateSchema)->ResumeStateSchema:
    return {}
def route_applicant(state:ResumeStateSchema)->Literal["respond_shortlist","respond_request_info","respond_rejection","respond_soft_rejection"]:
    tone=state["tone"]
    fit=state["fit"]
    flag=state["flag"]

    if flag=="major":
        return "respond_rejection"
    elif fit=="high" and tone in ["confident", "humble"] and flag in ["none", "minor"]:
        return "respond_shortlist"
    elif fit=="medium" and tone in ["confident","humble","desperate"] and flag in ["none",'minor']:
        return "respond_request_info"
    elif tone=='arrogant':
        return "respond_soft_rejection"
    elif fit=="low":    
        return "respond_soft_rejection"
    elif fit=="medium" and flag=="major":
        return "respond_rejection"
    else:
        return "respond_soft_rejection"
        
def respond_shortlist(state:ResumeStateSchema)->ResumeStateSchema:
    name=state["name"]
    email=state["email"]
    fit=state["fit"]
    fit_summary=state["fit_summary"]
    tone=state["tone"]
    tone_summary=state["tone_summary"]
    flag=state["flag"]
    flag_summary=state["flag_summary"]
    prompt=PromptTemplate(
        template="""
    You are an AI recruiter assistant. You will be provided with an evaluation of a job applicant's profile, including their qualification fit, communication tone, and any potential red flags.
    Your task is to write a respectful, warm, and professional email response informing the applicant that they have been shortlisted for the next stage of the recruitment process.
    Recruiter's Information:
    - Name of Recruiter: {recruiter_name}
    - Email of Recruiter: {recruiter_email}
    - Company's Name at which Recruiter works: {company_name}
    Use the following information:
    - Name of Applicant: {name}
    - Email of Applicant: {email}
    - Fit: {fit}
    - Fit Summary: {fit_summary}
    - Tone: {tone}
    - Tone Summary: {tone_summary}
    - Red Flags: {flag}
    - Flag Summary: {flag_summary}
    Craft a short, polite message congratulating the candidate, acknowledging their strengths, and expressing that they will be contacted with further details.
    **Below Best Regards add each at one line**
    - Name of Recruiter
    - Email of Recruiter
    - Company's Name
    """, input_variables=["name","email","fit","fit_summary","tone","tone_summary","flag","flag_summary","recruiter_name","recruiter_email","company_name"])
    chain=prompt|model_turbo
    response=chain.invoke({
        "name":name,
        "email":email,
        "fit":fit,
        "fit_summary":fit_summary,
        "tone":tone,
        "tone_summary":tone_summary,
        "flag":flag,
        "flag_summary":flag_summary,
        "recruiter_name":state["recruiter_name"],
        "recruiter_email":state['recruiter_email'],
        "company_name":state['company_name']
    })
    return {
        "response":response.content
    }

def respond_request_info(state:ResumeStateSchema)->ResumeStateSchema:
    name=state["name"]
    email=state["email"]
    fit=state["fit"]
    fit_summary=state["fit_summary"]
    tone=state["tone"]
    tone_summary=state["tone_summary"]
    flag=state["flag"]
    flag_summary=state["flag_summary"]
    prompt=PromptTemplate(
        template="""
    You are an AI recruiter assistant. You will be provided with an evaluation of a job applicant's profile, including their qualification fit, communication tone, and any potential red flags.
    Your task is to write a respectful and professional response message to the applicant, informing them that while their profile shows potential, additional information is required before a final decision can be made.
    Recruiter's Information:
    - Name of Recruiter: {recruiter_name}
    - Email of Recruiter: {recruiter_email}
    - Company's Name at which Recruiter works: {company_name}
    Use the following information:
    - Name of Applicant: {name}
    - Email of Applicant: {email}
    - Fit: {fit}
    - Fit Summary: {fit_summary}
    - Tone: {tone}
    - Tone Summary: {tone_summary}
    - Red Flags: {flag}
    - Flag Summary: {flag_summary}
    Your response should:
    - Politely request clarification, a portfolio, additional experience
    - Thank them for applying and express interest in learning more
    Below Best Regards add each at one line
    - Name of Recruiter
    - Email of Recruiter
    - Company's Name
    """, input_variables=["name","email","fit","fit_summary","tone","tone_summary","flag","flag_summary","recruiter_name","recruiter_email","company_name"])
    chain=prompt|model_turbo
    response=chain.invoke({
        "name":name,
        "email":email,
        "fit":fit,
        "fit_summary":fit_summary,
        "tone":tone,
        "tone_summary":tone_summary,
        "flag":flag,
        "flag_summary":flag_summary,
        "recruiter_name":state["recruiter_name"],
        "recruiter_email":state['recruiter_email'],
        "company_name":state['company_name']
    })
    return {
        "response":response.content
    }

def respond_soft_rejection(state:ResumeStateSchema)->ResumeStateSchema:
    name=state["name"]
    email=state["email"]
    fit=state["fit"]
    fit_summary=state["fit_summary"]
    tone=state["tone"]
    tone_summary=state["tone_summary"]
    flag=state["flag"]
    flag_summary=state["flag_summary"]
    prompt=PromptTemplate(
        template="""
    You are an AI recruiter assistant. You will be provided with an evaluation of a job applicant's profile, including their qualification fit, communication tone, and any potential red flags.
    Your task is to write a respectful and professional response message to the applicant, informing them that they are not good fit right now but we appriciate their interest and encourage them to apply in future.
    Use the following information:
    - Name of Applicant: {name}
    - Email of Applicant: {email}
    - Fit: {fit}
    - Fit Summary: {fit_summary}
    - Tone: {tone}
    - Tone Summary: {tone_summary}
    - Red Flags: {flag}
    - Flag Summary: {flag_summary}
    Recruiter's Information:
    - Name of Recruiter: {recruiter_name}
    - Email of Recruiter: {recruiter_email}
    - Company's Name at which Recruiter works: {company_name}
    Below Best Regards add each at one line
    - Name of Recruiter
    - Email of Recruiter
    - Company's Name
    """, input_variables=["name","email","fit","fit_summary","tone","tone_summary","flag","flag_summary","recruiter_name","recruiter_email","company_name"])
    chain=prompt|model_turbo
    response=chain.invoke({
        "name":name,
        "email":email,
        "fit":fit,
        "fit_summary":fit_summary,
        "tone":tone,
        "tone_summary":tone_summary,
        "flag":flag,
        "flag_summary":flag_summary,
        "recruiter_name":state["recruiter_name"],
        "recruiter_email":state['recruiter_email'],
        "company_name":state['company_name']
    })
    return {
        "response":response.content
    }
def respond_rejection(state:ResumeStateSchema)->ResumeStateSchema:
    name=state["name"]
    email=state["email"]
    fit=state["fit"]
    fit_summary=state["fit_summary"]
    tone=state["tone"]
    tone_summary=state["tone_summary"]
    flag=state["flag"]
    flag_summary=state["flag_summary"]
    prompt=PromptTemplate(
        template="""
    You are an AI recruiter assistant. You will be provided with an evaluation of a job applicant's profile, including their qualification fit, communication tone, and any potential red flags.
    Your task is to write a respectful and professional response message to the applicant, informing them that there application has not been accepted for further consideration. 
    Don't suggest the possibility of reapplying or following up.
    Use the following information:
    - Name of Applicant: {name}
    - Email of Applicant: {email}
    - Fit: {fit}
    - Fit Summary: {fit_summary}
    - Tone: {tone}
    - Tone Summary: {tone_summary}
    - Red Flags: {flag}
    - Flag Summary: {flag_summary}
    Write a brief, clear rejection message that expresses appreciation for their interest but we are not movin forward with that.
    Recruiter's Information:
    - Name of Recruiter: {recruiter_name}
    - Email of Recruiter: {recruiter_email}
    - Company's Name at which Recruiter works: {company_name}
    Below Best Regards add each at one line
    - Name of Recruiter
    - Email of Recruiter
    - Company's Name
    """, input_variables=["name","email","fit","fit_summary","tone","tone_summary","flag","flag_summary","recruiter_name","recruiter_email","company_name"])
    chain=prompt|model_turbo
    response=chain.invoke({
        "name":name,
        "email":email,
        "fit":fit,
        "fit_summary":fit_summary,
        "tone":tone,
        "tone_summary":tone_summary,
        "flag":flag,
        "flag_summary":flag_summary,
        "recruiter_name":state["recruiter_name"],
        "recruiter_email":state['recruiter_email'],
        "company_name":state['company_name']
    })
    return {
        "response":response.content
    }

graph=StateGraph(ResumeStateSchema)

graph.add_node("get_applicant_info",get_applicant_info)
graph.add_node("analyze_tone",analyze_tone)
graph.add_node("analyze_fit",analyze_fit)
graph.add_node("analyze_flag",analyze_flag)
graph.add_node("route_node",route_node)
graph.add_node("respond_shortlist",respond_shortlist)
graph.add_node("respond_request_info",respond_request_info)
graph.add_node("respond_soft_rejection",respond_soft_rejection)
graph.add_node("respond_rejection",respond_rejection)

graph.add_edge(START,"get_applicant_info")
graph.add_edge(START,"analyze_tone")
graph.add_edge(START,"analyze_fit")
graph.add_edge(START,"analyze_flag")
graph.add_edge("get_applicant_info","route_node")
graph.add_edge("analyze_tone","route_node")
graph.add_edge("analyze_fit","route_node")
graph.add_edge("analyze_flag","route_node")
graph.add_conditional_edges("route_node",route_applicant)

graph.add_edge("respond_shortlist",END)
graph.add_edge("respond_request_info",END)
graph.add_edge("respond_soft_rejection",END)
graph.add_edge("respond_rejection",END)

workflow=graph.compile()
if st.button("Perform AI Screening"):
    with st.spinner("Analyzing Resume... Please wait..."):
        response=workflow.invoke({
            "resume":resume_text,
            "job_description":job_description
        })

    st.success("Screenig Completed")
    st.markdown("### AI Response")
    st.markdown(response['response'])