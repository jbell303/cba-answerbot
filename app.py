import openai
import streamlit as st
from streamlit_chat import message
from EmbeddingQuery import EmbeddingQuery as eq

# openai api key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# retrieve file contents
EMBEDDINGS_PATH = "s3://cbaanswerbot/cba_2015_base.csv"

# Setting page title and header
st.set_page_config(page_title="Fred-e", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Fred-E 🤖</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>A chatbot that answers questions about the FDX 2015 CBA</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by OpenAI's GPT-3.5 and GPT-4</p>", unsafe_allow_html=True)
st.markdown("<h3 >You can ask me questions like:</h3>", unsafe_allow_html=True)
st.markdown("<ul>\
                <li>What types of deadhead flights qualify for first class travel?</li>\
                <li>When do I have to complete final deviation check-in if my first operating leg is OAK to HNL?</li>\
                <li>If I have 7 days of vacation and the r-day value is 4:36, how many days can I extend my vacation on a reserve line?</li>\
                <li>Explain in simple terms what happens if my trip is cancelled for operational reasons prior to block out.</li>\
            </ul>", unsafe_allow_html=True)
st.markdown("<h3 >Tips:</h3>", unsafe_allow_html=True)
st.markdown("<ul>\
                <li>Try to be as specific as possible</li>\
                <li>Use similar phrasing to the contract when possible</li>\
                <li>Try re-phrasing the question with more detail if unsuccessful on the first attempt</li>\
                <li>Use the sidebar to clear the conversation or change the model</li>\
                <li>Select GPT-4 on the sidebar if the question requires complex reasoning</li>\
            </ul>", unsafe_allow_html=True)

# Set org ID and API key
# openai.organization = "<YOUR_OPENAI_ORG_ID>"
# openai.api_key = "<YOUR_OPENAI_API_KEY>"

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You answer questions about the fedex pilot contract."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    # create a new query object
    query = eq(prompt, EMBEDDINGS_PATH, gpt_model=model)
    st.session_state['messages'].append({"role": "user", "content": query.query_message()})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
