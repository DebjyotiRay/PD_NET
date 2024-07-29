import streamlit as st
from openai import OpenAI
import re
# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Streamlit application
st.title("Gene Citation Finder")

# User input
user_input = st.text_input("Enter your biomedical question:", "What are the genes responsible for Parkinson's disease?")

# Prompt to emphasize specificity and citations
prompt = (
    f"Answer the following biomedical question in a very specific manner, "
    f"providing only the names of the genes or causes when asked. Do not explain anything extra "
    f"unless specifically asked in the user query. Provide citations to support your answer, "
    f"included with links. Highlight only the main keywords or genes:\n\n{user_input}"
)

# Button to submit the input and get the response
if st.button("Get Genes and Citations"):
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    # Display the response as plain text
    st.write("Response")
    st.write(response)

    # Extract and format citations
    citations = re.findall(r'\[(.*?)\]\((.*?)\)', response)
    if citations:
        st.write("Citations:")
        for text, link in citations:
            st.write(f"[{text}]({link})")
