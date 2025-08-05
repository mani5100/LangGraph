from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import Annotated,TypedDict
import operator
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
import streamlit as st
from streamlit_quill import st_quill

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #4CAF50;'>✍️ Blog Evaluator</h1>
        <p style='font-size: 18px; color: #555;'>Get detailed feedback and scoring for your blog across clarity, engagement, and depth.</p>
    </div>
    """,
    unsafe_allow_html=True
)

load_dotenv()


model=ChatOpenAI(model="gpt-4o-mini")
class EvaluateSchema(BaseModel):
    feedback:Annotated[str,Field(description="Detailed Feedback of blog")]
    score:Annotated[int,Field(description="This is the score out of 10",gt=0,le=10)]
StructuredModel=model.with_structured_output(EvaluateSchema)


class BlogEvaluationState(TypedDict):
    blog:str

    clarity_structure_feedback:str
    engagement_value_feedback:str
    credibility_depth_feedback:str
    overall_feedback:str

    individual_score:Annotated[list[int],operator.add]
    final_score:float


graph=StateGraph(BlogEvaluationState)


def gen_clarity_structure_feedback(state:BlogEvaluationState)->BlogEvaluationState:
    blog=state["blog"]
    template=PromptTemplate(
        template="""You have to act as a professional Content Reviewer. You will be given the blog and you have to check only for the clarity and structure of the blog.
        You have to check Following thing to ensure Clarity and Structure.
        1. A compelling headline
        2. Clear introduction, body, and conclusion
        3. Well-organized sections
    Based on that you have to give appropraite feedback and score between 1 to 10. The blog is given below:
    Blog: {blog}
    """,
    input_variables=["blog"]
    )
    prompt=template.invoke({"blog":blog})
    response=StructuredModel.invoke(prompt)
    return {
        "clarity_structure_feedback":response.feedback,
        "individual_score":[response.score]
    }
graph.add_node("gen_clarity_structure_feedback",gen_clarity_structure_feedback)


def gen_engagement_value_feedback(state:BlogEvaluationState)->BlogEvaluationState:
    blog=state["blog"]
    template=PromptTemplate(
        template="""You have to act as a professional Content Reviewer. You will be given the blog and you have to check only for the Engagement and Value of the blog.
        You have to check Following thing to ensure Engagement and Value. 
        1. Solves a problem, answers a question, or offers a fresh perspective
        2. Uses examples, or storytelling to maintain engagement
        3. Has a conversational or audience-appropriate tone
    Based on that you have to give appropraite feedback and score between 1 to 10. The blog is given below:
    Blog: {blog}
    """,
    input_variables=["blog"]
    )
    prompt=template.invoke({"blog":blog})
    response=StructuredModel.invoke(prompt)
    return {
        "engagement_value_feedback":response.feedback,
        "individual_score":[response.score]
    }
graph.add_node("gen_engagement_value_feedback",gen_engagement_value_feedback)


def gen_credibility_depth_feedback(state:BlogEvaluationState)->BlogEvaluationState:
    blog=state["blog"]
    template=PromptTemplate(
        template="""You have to act as a professional Content Reviewer. You will be given the blog and you have to check only for the Credability and Depth of the blog.
        You have to check Following thing to ensure Credibility and Depth.
        1. Includes accurate information or data
        2. Shows depth of understanding, not just surface-level content
    Based on that you have to give appropraite feedback and score between 1 to 10. The blog is given below:
    Blog: {blog}
    """,
    input_variables=["blog"]
    )
    prompt=template.invoke({"blog":blog})
    response=StructuredModel.invoke(prompt)
    return {
        "credibility_depth_feedback":response.feedback,
        "individual_score":[response.score]
    }
graph.add_node("gen_credibility_depth_feedback",gen_credibility_depth_feedback)


def gen_overall_feedback(state:BlogEvaluationState)->BlogEvaluationState:
    clarity_structure_feedback=state["clarity_structure_feedback"]
    engagement_value_feedback=state["engagement_value_feedback"]
    credibility_depth_feedback=state["credibility_depth_feedback"]
    template=PromptTemplate(
        template="""You will be given 3 different feedbacks of a Blog. One is Clarity and Structure Feedback, Second is Engagement and Value Feedback and Third is Credibility and Depth Feedack.
        You have to write a summerized feedback that tell the issues and measures that a content writer can take to improve that blog.
        The feedbacks are as follows:
        Clarity and Structure Feedback: {clarity_structure_feedback}
        Engagement and Value Feedback:  {engagement_value_feedback}
        Credibility and Depth Feedack:  {credibility_depth_feedback}
    """,
    input_variables=["clarity_structure_feedback","engagement_value_feedback","credibility_depth_feedback"]
    )
    prompt=template.invoke({
        "clarity_structure_feedback":clarity_structure_feedback,
        "engagement_value_feedback":engagement_value_feedback,
        "credibility_depth_feedback":credibility_depth_feedback
    })
    response=model.invoke(prompt)
    score=(sum(state["individual_score"]))/len(state['individual_score'])
    return {
        "overall_feedback":response.content,
        "final_score":score
    }
graph.add_node("gen_overall_feedback",gen_overall_feedback)


graph.add_edge(START,"gen_clarity_structure_feedback")
graph.add_edge(START,"gen_engagement_value_feedback")
graph.add_edge(START,"gen_credibility_depth_feedback")

graph.add_edge("gen_clarity_structure_feedback","gen_overall_feedback")
graph.add_edge("gen_engagement_value_feedback","gen_overall_feedback")
graph.add_edge("gen_credibility_depth_feedback","gen_overall_feedback")

graph.add_edge("gen_overall_feedback",END)


workflow=graph.compile()


st.markdown("### 📝 Write or Paste Your Blog Below")
blog = st_quill(placeholder="Start writing your blog here...")


if blog: 
    if st.button("🚀 Evaluate Blog", use_container_width=True):
        # optional toast
        st.toast("Scoring started...", icon="🧠")

        with st.spinner("Evaluating your blog... please wait ⏳"):
            final_state = workflow.invoke({"blog": blog})
            score = final_state['final_score']

        st.success("✅ Evaluation Complete")
        st.markdown(f"## 📊 Final Score: `{round(score, 2)} / 10`")
        st.divider()

        # feedback card
        st.markdown(
            """
            <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
                <h3 style='margin-top: 0; color:#000000'>🗣️ Overall Feedback</h3>
            """,
            unsafe_allow_html=True
        )
        st.write(final_state.get("overall_feedback", "No feedback provided."))
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div style='text-align:center; margin-top:40px;'>
            <p style='font-size:18px; color:#888;'>
                ✍️ Please write or paste your blog above to unlock the evaluation button.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )