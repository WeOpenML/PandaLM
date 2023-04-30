import os
import sys
import re
import fire
import gradio as gr
import torch
import transformers
import traceback

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from queue import Queue
from threading import Thread



DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def build_prompt(instruction, input, resp1, resp2, result=None, explain=None, ref=None):

    rsp = f'### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}'

    if input:
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{rsp}\n\n### Evaluation:\n"
    else:
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n{rsp}\n\n### Evaluation:\n"

    if result:
        output_sequence = f'{result}\n\n### Reason: {explain}\n\n### Reference: {ref}\n'
        return input_sequence, output_sequence
    else:
        return input_sequence


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def post_process_output(text):
    text = text.strip().split('### Evaluation:')[1].strip()
    pattern = re.compile(r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]")
    pattern.sub("", text.strip()).strip()
    return text


def main(
    load_8bit: bool = False,
    base_model: str = 'WeOpenML/PandaLM-7B-v1',
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    server_port: int = 31228,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (base_model), "Please specify a --base_model, e.g. --base_model='WeOpenML/PandaLM-7B-v1'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
#        torch_dtype=torch.float16,
        device_map="auto",
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        response1=None,
        response2=None,
        temperature=0,
        top_p=1,
        top_k=1,
        num_beams=4,
        max_new_tokens=150,
        stream_output=True,
        repetition_penalty=1.2,
        **kwargs,
    ):
        # prompt = prompter.generate_prompt(instruction, input)
        prompt = build_prompt(instruction, input, response1, response2)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                early_stopping=True,
                repetition_penalty=repetition_penalty,
                **kwargs
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield post_process_output(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield post_process_output(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=3,
                label="Instruction",
                value="Build a Java program to output the following message",
            ),
            gr.components.Textbox(lines=2, label="Input", value="Hello World!"),
            gr.components.Textbox(lines=4, label="Response 1", value='System.out.println("Hello World");'),
            gr.components.Textbox(lines=4, label="Response 2", value='public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println("Hello World!");\n    }\n}'),
            
            gr.components.Slider(
                minimum=0, maximum=1, value=0, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=1, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=1, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=8, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=1024, step=1, value=512, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output", value=True),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Which response is better?",
            )
        ],
        title="PandaLM",
        description="Compare different responses with a given context.",  # noqa: E501
    ).queue().launch(server_name=server_name, share=share_gradio, server_port=server_port)

if __name__ == "__main__":
    fire.Fire(main)
