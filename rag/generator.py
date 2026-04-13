import torch
from transformers import pipeline

HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

generator = pipeline(
    "text-generation",
    model=HF_MODEL,
    torch_dtype=dtype,
    device=device
)


def generate_answer(system_prompt, user_prompt):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    output = generator(
        messages,
        max_new_tokens=500,
        do_sample=False,
        temperature=0.2,
        return_full_text=False
    )

    if isinstance(output, list) and len(output) > 0:
        item = output[0]

        if "generated_text" in item:
            generated = item["generated_text"]

            if isinstance(generated, list) and len(generated) > 0:
                last_msg = generated[-1]

                if isinstance(last_msg, dict) and "content" in last_msg:
                    return last_msg["content"].strip()

    return str(output)