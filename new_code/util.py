import openai
import os

openai.base_url = "https://api.rcouyi.com/v1/"

def load_message(prompt_content: str, system_prompt: str = "hello, what can i help you?"):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_content}
    ]
    return messages


def call_llm(
    messages: list,
    api_key,
    model_name,
    temperature: float = 0,
    max_tokens: int = 16384,
):

    openai.api_key = api_key

    response = openai.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


    if response.choices[0].message.content:
        output = response.choices[0].message.content.strip()
        return output
    else:
        return None
