import multiprocessing
import os
import re
import signal
import sys
from dataclasses import replace

from tqdm import tqdm
from watermark_benchmark.utils.classes import Generation

prompt_cot = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
Your evaluation hinges on the following factors: accuracy, level of details, typographical correctness, grammatical correctness and lexical correctness.
Be as objective as possible, and account for every one of these criteria in our evaluation. Begin your evaluation by providing a short explanation. 
After providing your explanation, please rate the response on a scale of 1 to 100 by strictly following this format: "[[rating]]", for example: "Rating: [[51]]".
Your rating must come AFTER your explanation, not before. <|eot_id|> 
<|start_header_id|>user<|end_header_id|> 
Prompt:\n{}\nResponse:\n{} <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"""

prompt_no_cot = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. 
Remove points as soon as one of the criteria is missed.
<|start_header_id|>user<|end_header_id|> 
Prompt:\n{}\nResponse:\n{} <|eot_id|> <|start_header_id|>assistant<|end_header_id|> Grade: """


def writer_process(queue, config, w_count):
    from watermark_benchmark.utils import get_output_file, setup_randomness

    setup_randomness(config)
    outfilepath = get_output_file(config)

    for _ in tqdm(range(w_count), total=w_count, desc="Rating"):
        task = queue.get(block=True)
        if task is None:
            queue.put(None)
            return

        with open(outfilepath, "a") as outfile:
            outfile.write("\n".join(str(gen) for gen in task) + "\n")


def rating_process(config, generations, writer_queue, device):
    """
    Runs the model on the given generations and calculates the rating for each generation.

    Args:
        config (Config): Configuration object.
        generations (List[Generation]): List of generations to rate.
        writer_queue (Queue): Queue to write the rated generations to.
        device (int): Index of the GPU device to use.

    Returns:
        None
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Imports
    import torch
    from watermark_benchmark.servers import get_model
    from watermark_benchmark.utils import get_server_args, setup_randomness

    torch.set_num_threads(1)

    setup_randomness(config)

    # Setup server
    config.model = "meta-llama/Meta-Llama-3-8B-Instruct"  # "meta-llama/Meta-Llama-3-8B-Instruct"
    config.max_new_tokens = 4
    config.dtype = "float16"
    config.num_return_sequences = 1
    inference_engine = config.engine
    server = get_model(inference_engine, config, **get_server_args(config))
    tokenizer = server.tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    tasks = []
    for generation in generations:
        tasks.append(
            prompt_no_cot.format(
                generation.prompt.replace("[/INST]", "")
                .replace("[INST]", "")
                .replace("<<SYS>>", "")
                .replace("<</SYS>>", "")
                .strip(),
                generation.response,
            )
        )

    # Clip sequences that are too long
    max_token_length = 8000
    for i in tqdm(range(len(tasks)), total=len(tasks), desc="Encoding"):
        task = tasks[i]
        if len(task) > max_token_length:
            encoded_task = tokenizer(task)["input_ids"]
            if len(encoded_task) > max_token_length:
                print(
                    "Warning: Task too long ({} tokens), clipping to {} tokens".format(
                        len(encoded_task), max_token_length
                    )
                )
                task = tokenizer.decode(encoded_task[:max_token_length])
        tasks[i] = task

    print("Encoding done. Ready for rating.")

    # Run model
    outputs = server.run(tasks, config, 0.0, use_tqdm=True)
    num_regex = re.compile(r"([0-9]+\.*[0-9]*)(/100)?")

    # Parse outputs
    for idx, gen in enumerate(outputs):
        try:
            raw = 0.0
            matches = re.findall(num_regex, gen.response)
            if matches and len(matches):
                val = matches[-1][0].replace("[", "").replace("]", "")
                if "/" in val:
                    raw = float(val.split("/")[0]) / float(val.split("/")[1])
                else:
                    raw = float(val) / 100

            raw = max(min(raw, 1), 0)
        except Exception as e:
            generations[idx] = replace(generations[idx], rating=-1)
            print("Encountered error while parsing rating: {}".format(e))
            continue

        if idx >= len(generations):
            print(
                "Warning: Received more outputs than generations ({} vs {})".format(
                    len(outputs), len(generations)
                )
            )
            break
        generations[idx] = replace(generations[idx], rating=raw)

    # Write to file
    writer_queue.put(generations)


def run(config_file, generations=None):
    from watermark_benchmark.utils import (
        get_input_file,
        get_output_file,
        load_config,
        setup_randomness,
    )

    # load config
    config = (
        load_config(config_file) if type(config_file) == str else config_file
    )
    setup_randomness(config)

    # load generations
    generations = (
        Generation.from_file(get_input_file(config))
        if not generations
        else generations
    )
    outfilepath = get_output_file(config)
    if not os.path.exists(outfilepath):
        Generation.to_file(outfilepath)
    existing = {
        str(
            (
                (
                    g.watermark.to_dict(True, True)
                    if g.watermark is not None
                    else g.temp
                ),
                g.id,
                g.attack,
            )
        )
        for g in Generation.from_file(outfilepath)
    }
    tasks = [
        g
        for g in generations
        if str(
            (
                (
                    g.watermark.to_dict(True, True)
                    if g.watermark is not None
                    else g.temp
                ),
                g.id,
                g.attack,
            )
        )
        not in existing
    ]

    if not len(tasks):
        return

    ct = 1 + (len(tasks) // len(config.get_devices()))
    global_manager = multiprocessing.Manager()
    processes = []
    writer_queue = global_manager.Queue()

    for idx, device in enumerate(config.get_devices()):
        local_gens = tasks[idx * ct : (idx + 1) * ct]
        processes.append(
            multiprocessing.Process(
                target=rating_process,
                args=(config, local_gens, writer_queue, device),
            )
        )
        processes[-1].start()

    writer = multiprocessing.Process(
        target=writer_process,
        args=(writer_queue, config, len(config.get_devices())),
    )
    writer.start()

    # Setup signal handler
    def graceful_exit(sig, frame):
        print("Stopping all processes...")
        for p in processes:
            p.terminate()
        writer.terminate()
        exit()

    signal.signal(signal.SIGINT, graceful_exit)

    writer.join()
    for p in processes:
        p.terminate()

    print("Finished all tasks, exiting")
    # graceful_exit(None, None)


def main():
    multiprocessing.set_start_method("spawn")
    run(sys.argv[1])