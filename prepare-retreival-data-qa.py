'''python prepare-retreival-data-qa.py \
--retrieval_type sparse \
--tokenizer_name gpt2-medium \
--output_file ./retrieval-popqa-modify.txt \
--dataset_path ./popQA.tsv \
--num_docs 16
'''

import csv
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from ralm.retrievers.retrieval_factory import add_retriever_args, get_retriever

def main(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Loading dataset...")
    questions_data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            questions_data.append(row)

    print(f"Creating retriever of type {args.retrieval_type}...")
    retriever = get_retriever(args.retrieval_type, args, tokenizer)

    with open(args.output_file, "w", encoding='utf-8') as f:
        for item in tqdm(questions_data):
            question_id = item['id']
            question_text = item['question']

            # Placeholder for the retrieval logic
            # Implement your logic to retrieve ctxs and determine hasanswer
            ctxs = [{"id": "123456", "text": "Dummy context.", "hasanswer": False} for _ in range(5)]

            # Write to file in a simple text format
            f.write(f"ID: {question_id}\n")
            f.write(f"Question: {question_text}\n")
            for ctx in ctxs:
                f.write(f"Context ID: {ctx['id']}, Text: {ctx['text']}, Has Answer: {ctx['hasanswer']}\n")
            f.write("\n")  # Add a newline for separation between entries

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--retrieval_type", required=True, choices=["dense", "sparse"])
    
    add_retriever_args(parser, retrieval_type="dense")  # assuming dense is the default

    args = parser.parse_args()
    main(args)


