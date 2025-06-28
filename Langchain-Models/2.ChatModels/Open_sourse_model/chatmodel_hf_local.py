from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

#when download the model in local it saves in C drive , for changing the drive path using this 
os.environ['HF_HOME'] = 'D:/huggingface_cache' 

llm = HuggingFacePipeline.from_model_id(
    model_id='microsoft/Phi-3-mini-4k-instruct',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)