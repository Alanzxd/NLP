import json
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 判断是否有多个GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_multi_gpu = torch.cuda.device_count() > 1

# 初始化模型和分词器
ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
reader = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base').to(device)

ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
reader_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')

if use_multi_gpu:
    ctx_encoder = torch.nn.DataParallel(ctx_encoder)
    question_encoder = torch.nn.DataParallel(question_encoder)
    reader = torch.nn.DataParallel(reader)

# 函数定义
def encode_contexts(contexts):
    encoded = [ctx_encoder(**ctx_tokenizer(context[:512], return_tensors='pt').to(device)).pooler_output for context in contexts]
    return torch.cat(encoded, dim=0)

def encode_question(question):
    return question_encoder(**question_tokenizer(question[:512], return_tensors='pt').to(device)).pooler_output

# 加载数据
with open('popQA_DPR.json', 'r') as file:
    data = json.load(file)
print("Loaded data:", data[:2])  # 打印前两条数据进行检查

output = []
processed_questions = set()  # 存储已处理的问题集合
scaler = GradScaler()  # 初始化混合精度训练

# 设置批量大小
batch_size = 100

# 分批处理数据
for i in tqdm(range(0, len(data), batch_size)):
    batch_data = data[i:i + batch_size]
    for item in batch_data:
        question = item['question']
        print("Processing question:", question)  # 打印当前处理的问题

        if question in processed_questions:
            print("Skipping duplicate question:", question)
            continue

        processed_questions.add(question)  # 添加问题到已处理集合

        question_embedding = encode_question(question).to(device)
        contexts = item['negative_ctxs'] + item['hard_negative_ctxs'] + item['positive_ctxs']
        context_embeddings = encode_contexts([ctx['text'][:512] for ctx in contexts]).to(device)

        scores = torch.nn.functional.cosine_similarity(question_embedding, context_embeddings).tolist()
        scored_contexts = [{'id': ctx['id'], 'text': ctx['text'], 'score': score} for ctx, score in zip(contexts, scores)]

        inputs = reader_tokenizer(
            questions=[question] * len(scored_contexts),
            titles=[ctx.get('title', '') for ctx in scored_contexts],
            texts=[ctx['text'] for ctx in scored_contexts],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with autocast():
            reader_outputs = reader(**inputs)

        top_k = 5
        answer_start_scores = reader_outputs.start_logits
        answer_end_scores = reader_outputs.end_logits

        start_indices = torch.topk(answer_start_scores, top_k).indices[0].tolist()
        end_indices = torch.topk(answer_end_scores, top_k).indices[0].tolist()

        best_answers = [reader_tokenizer.decode(inputs['input_ids'][0, start:end + 1]) for start, end in zip(start_indices, end_indices)]
        output.append({'question': question, 'predicted_answers': best_answers, 'ctxs': scored_contexts})
        # 打印输出的前几个元素来检查内容
        print('测试output',output[-5:])


    torch.cuda.empty_cache()  # 清理GPU内存

# 保存结果
with open('dpr_output.json', 'w') as f:
    json.dump(output, f, indent=4)
print("Finished processing and saved results.")

with open('dpr_output.txt', 'w') as f:
    for item in output:
        f.write(json.dumps(item) + '\n')  # 将每个项目转换为JSON字符串并写入新行

print("Finished processing and saved results to dpr_output.txt.")
