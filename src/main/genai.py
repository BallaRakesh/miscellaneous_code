import os
import google.generativeai as palm
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
import streamlit as st

# Set the PaLM API key
os.environ['PALM_API_KEY'] = 'api key'

# TyuIOtSPl90OPUYAGgG1xUY1_IxGDY5ATHrTWy80yyHyiPzxy

# Configure the model
palm.configure(api_key=os.environ['PALM_API_KEY'])
#calling
# # Use the model
# response = palm.generate_text(
#     model='models/text-bison-001',
#     prompt="I want to open a restaurant for Italian",
#     temperature=0.6,
#     max_output_tokens=100
# )

# print(response.result)

# Custom LLM class for PaLM
class PalmLLM(LLM):
    model: str = "models/text-bison-001"
    temperature: float = 0.6
    max_output_tokens: int = 100

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"Calling PaLM API with prompt: {prompt}")
        response = palm.generate_text(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )
        # print(f"Response: {response.result}")
        return response.result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature, "max_output_tokens": self.max_output_tokens}

    @property
    def _llm_type(self) -> str:
        return "palm"

# Initialize the custom PalmLLM
llm = PalmLLM()
###############################

def calling_llm(cuisine_):
    
    prompt_template_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = 'give me the name of the restaurant for the cuisine, mention only restaurant name {cuisine}.'
    )
    # prompt_template_name.format(cuisine = 'italian')

    name_chain = LLMChain(llm = llm, prompt = prompt_template_name, output_key = "restaurant_name")
    ########### ***************** ###########
    # result = name_chain.run(cuisine="Italian")
    # print(result)
    ################################################
    prompt_template_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template = 'suggest some 10 items and items should be seperated with comma for the given restaurant{restaurant_name}.'
    )
    # prompt_template_name.format(cuisine = 'italian')
    food_item_chain = LLMChain(llm = llm, prompt = prompt_template_items, output_key = "menu_items_name")
    ########### ***************** ###########
    # result = food_item_chain.run(restaurant_name = "starbugs")
    # print(result)
    #################################################
    #using SimpleSequentialChain
    ################################################
    # chain = SimpleSequentialChain(chains = [name_chain, food_item_chain])
    # responce = chain.run('Indian')
    # print(responce)
    ###################################################
    #using SequentialChain
    chain = SequentialChain(
        chains = [name_chain, food_item_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items_name']
    )
    final_responce = chain({'cuisine': cuisine_})
    print(final_responce)
    return final_responce



# streamlit
st.title('Restaurant Name Generator')
cuisine = st.sidebar.selectbox("pick a cuisine", ('Indian', 'Italian', 'spanish'))
def generate_responce(cuisine):
    return calling_llm(cuisine)

if cuisine:
    responce_ = generate_responce(cuisine)
    st.header(responce_['restaurant_name'].strip())
    menu_items = responce_['menu_items_name'].strip().split(',')
    st.write('**MENU ITEMS**')
    for idx, items in enumerate(menu_items):
        st.write(idx+1, ')', items.strip())