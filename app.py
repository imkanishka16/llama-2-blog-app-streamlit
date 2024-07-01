import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import CTransformers
from transformers import AutoTokenizer
from langchain.chains import ConversationChain
import transformers
import torch
import warnings
warnings.filterwarnings('ignore')


#Function to get respose from Llama 2 model
def getLLamaresponse(input_text,no_words,blog_style):
    llm = CTransformers(model= 'llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        config = {
                            'max_new_tokens': 256,
                            'temperature':0.01
                        })
    #calling Llama 2 model
    # model="meta-llama/Llama-2-7b-chat-hf"
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model,
    #     task="text-generation",
    #     pipeline_kwargs=dict(
    #         max_new_tokens=512,
    #         do_sample=False,
    #     ),
    #     model_kwargs={'temperature':0.01}
    # )
    
    template = """
    write a blog for {blog_style}job profile for a topic {input_text} within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                            template=template)
    
    #Generate the response from the LLama 2 model
    respose = llm(prompt.format(blog_style= blog_style,input_text= input_text,no_words=no_words))
    print(respose)
    return respose  


st.set_page_config(page_title="Generate Blogs",
                   page_icon = "🥸",
                   layout="centered",
                   initial_sidebar_state = "collapsed")

st.header("Generate Blogs 🥸")

input_text = st.text_input("Enter the Blog Topic")

#creating to more columns for additional 2 fields
col1,col2 = st.columns([5,5])

with col1:
    no_words =  st.text_input("No of words")
with col2:
    blog_style = st.selectbox("Writing the blog for", ("Researcher","Data Scientist","Common People"),index=0)
    
submit = st.button("Generate")

#final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))
    
