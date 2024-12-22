from llama_cpp import Llama
from transformers import AutoTokenizer
from retrieval.retrieval_qdrant import retrieval_qdrant

def rag_ans(rag_func, question):
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Llama(
        model_path='models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
        n_ctx=512,
        n_gpu_layers=-1       # Number of model layers to offload to GPU
    )


    if rag_func == True:
        retrieval_result = retrieval_qdrant(question)
        PROMPT = f'''
        당신은 유용한 AI 보험설계사입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
        '''

        question_templete = f'''
        사용자의 질문에 참조할 정보는 아래와 같아.

        1. {retrieval_result[0]}
        2. {retrieval_result[1]}
        3. {retrieval_result[2]}

        사용자의 질문은 다음과 같다.

        {question}

        최종 답변은 500자로 한정해주고, 그 안에 완벽한 문장으로 마무리해줘
        '''

    else:
        PROMPT = f'''
        당신은 유용한 AI 보험설계사입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
        '''

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{question_templete}"}
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt=True
    )

    generation_kwargs = {
        "max_tokens":2048,
        "stop":["<|eot_id|>"],
        "top_p":0.9,
        "temperature":0.6,
        "echo":True, # Echo the prompt in the output
    }

    resonse_msg = model(prompt, **generation_kwargs)
    return resonse_msg['choices'][0]['text'][len(prompt):]