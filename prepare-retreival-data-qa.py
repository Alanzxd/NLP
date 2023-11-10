'''python prepare-retreival-data-qa.py \
--retrieval_type sparse \
--tokenizer_name gpt2-medium \
--output_file ./retrieval-popqa-modify.json \
--dataset_path ./popQA.tsv \
--num_docs 16
'''
print('test')
import json
import csv
import sys
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from ralm.file_utils import print_args
from ralm.retrievers.retrieval_factory import add_retriever_args, get_retriever

def check_answer_in_context(context, answers):
    # Placeholder for actual answer checking logic
    # You'll need to implement this based on your specific needs
    return any(answer.lower() in context.lower() for answer in answers)

def main(args):
    # Dump args
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Loading questions from CSV...")
    questions_data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions_data.append(row)

    print(f"Creating retriever of type {args.retrieval_type}...")
    retriever = get_retriever(args.retrieval_type, args, tokenizer)

    output_data = []
    for item in tqdm(questions_data):
        question = item['question']
        question_id = item['id']
        possible_answers = json.loads(item['possible_answers'].replace('""', '"'))

        # Tokenize the question for retrieval
        encoded_question = tokenizer(question, return_tensors='pt')

        # Retrieve contexts using the retriever
        # This part is a placeholder and needs to be implemented based on your retrieval system
        retrieved_contexts = retriever.retrieve(encoded_question.input_ids, args.num_docs)

        # Check if contexts contain the answer
        ctxs = []
        for ctx_id, context in enumerate(retrieved_contexts):
            has_answer = check_answer_in_context(context, possible_answers)
            ctxs.append({
                "id": ctx_id,  # This would actually be a unique identifier for the context
                "text": context,
                "hasanswer": has_answer
            })

        # Construct the output data structure
        output_data.append({
            "id": question_id,
            "question": question,
            "ctxs": ctxs
        })

    print(f"Writing to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print("Done!")

if __name__ == '__main__':
    assert len(sys.argv) > 1 and sys.argv[1] == "--retrieval_type"
    retrieval_type = sys.argv[2]
    assert retrieval_type in ["dense", "sparse"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")

    # Retrieval params
    parser.add_argument("--retrieval_type", required=True)
    parser.add_argument("--num_docs", type=int, default=5)  # Number of contexts to retrieve
    add_retriever_args(parser, retrieval_type)

    args = parser.parse_args()
    main(args)
