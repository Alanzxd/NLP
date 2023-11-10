import json
import sys
import argparse
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

from ralm.file_utils import print_args
from ralm.retrievers.retrieval_factory import add_retriever_args, get_retriever

RETRIEVAL_TYPES = ["dense", "sparse"]

def main(args):
    # Dump args
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Loading dataset...")
    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()

    # Tokenize the data
    print("Tokenizing dataset...")
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    # Set device for PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Creating retriever of type {args.retrieval_type}...")
    retriever = get_retriever(args.retrieval_type, args, tokenizer)

    # If retriever has a model and you want to use GPU for it
    if hasattr(retriever, 'model'):
        retriever.model.to(device)

    print("Processing dataset...")
    prev_end_loc = 0
    data = []
    for begin_loc in tqdm(range(0, dataset_len, args.stride)):
        end_loc = min(begin_loc + args.max_length, dataset_len)
        target_begin_loc = prev_end_loc

        # If your retriever.retrieve function expects tensors on a GPU
        input_ids = encodings.input_ids[0, target_begin_loc:end_loc].unsqueeze(0).to(device)
        retrieved_data = retriever.retrieve(input_ids, k=args.num_docs)

        d = {
            "begin_location": target_begin_loc,
            "end_location": end_loc,
            "future": tokenizer.decode(input_ids[0].to('cpu'))
        }

        data.append(d)
        prev_end_loc = end_loc

        if end_loc >= dataset_len:
            break

    print(f"Finished processing {len(data)}/{len(data)} strides")
    print(f"Writing to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print("Done!")


if __name__ == '__main__':
    assert sys.argv[1] == "--retrieval_type"
    retrieval_type = sys.argv[2]

    assert retrieval_type in RETRIEVAL_TYPES

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")

    # Model params
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=32)

    # Retrieval params
    parser.add_argument("--retrieval_type", required=True, choices=RETRIEVAL_TYPES)
    parser.add_argument("--num_docs", type=int, default=1)
    add_retriever_args(parser, retrieval_type)

    args = parser.parse_args()
    main(args)
