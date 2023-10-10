from langchain.schema import prompt_template
import openai
import langchain
import os


from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory

'''
def practice():
   prompt = PromptTemplate.from_template("what is capital of {place} ")
   llm = OpenAI(temperature=0.5)

   chain = LLMChain(llm=llm, prompt=prompt)

   city = "uttarakhand"
   output = chain.run(city)
   print(output)



#LLM to get name of an e commerce store

prompt = PromptTemplate.from_template("suggest the name of the e commerce store that sells {product}?")
llm =  OpenAI(temperature=1)
chain1 = LLMChain(llm = llm, prompt=prompt)
# product="iphone"
# output = chain.run(product)
# print(output)


#LLM to get name of prodcuts for that e commerce store

prompt = PromptTemplate.from_template("suggest some more products for this {store}")
llm =  OpenAI(temperature=1)
chain2 = LLMChain(llm = llm, prompt=prompt)
# store="amazon"
# output = chain.run(store)
# print(output)


#Create an overallchain from simple sequential chain
chain = SimpleSequentialChain(
   chains=[chain1, chain2], verbose=True
)
output = chain.run("Liqour")










#example of sequential chain
#chain1 This is an LLMChain to write a synopsis given a title of a play and the era it is set in.

llm = OpenAI(temperature=.7)
synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
synopsis_prompt_template = PromptTemplate(input_variables=["title", "era"], template=synopsis_template)
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template, output_key="synopsis")


#chain2 This is an LLMChain to write a review of a play given a synopsis.

llm = OpenAI(temperature=.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")


#created sequential chain

overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True)

print(overall_chain({"era": "Renaissance", "title": "The Tempest"}))









#Agents Demo
llm = OpenAI(temperature=1)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION)
output = agent.run("how old will ms dhoni be in 2029?")
print(output)


'''


#Memory in LLMs
llm =  OpenAI(temperature=1)
prompt = PromptTemplate.from_template("suggest the name of the e commerce store that sells {product}?")
chain = LLMChain(llm = llm, prompt=prompt, memory=ConversationBufferMemory())
output =  chain.run("fruits")
output =  chain.run("books")

print(chain.memory.buffer)
print(output)