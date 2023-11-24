import json
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): 包含数据的列表，每个元素是一个字典，表示一个数据样本。
                每个字典至少包含以下键：
                - 'question': 问题文本
                - 'negative_ctxs': 负面上下文的列表
                - 'hard_negative_ctxs': 困难负面上下文的列表
                - 'positive_ctxs': 正面上下文的列表
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 提取样本中的问题和上下文
        question = sample['question']
        negative_ctxs = sample['negative_ctxs']
        hard_negative_ctxs = sample['hard_negative_ctxs']
        positive_ctxs = sample['positive_ctxs']

        # 构建数据样本字典
        data_sample = {
            'question': question,
            'negative_ctxs': negative_ctxs,
            'hard_negative_ctxs': hard_negative_ctxs,
            'positive_ctxs': positive_ctxs
        }

        return data_sample

# 判断是否有多个GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.device_count() > 1:
    use_multi_gpu = True
else:
    use_multi_gpu = False

if use_multi_gpu:
    # 初始化多个GPU
    ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to('cuda:0')
    ctx_encoder = torch.nn.DataParallel(ctx_encoder)
    
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to('cuda:0')
    question_encoder = torch.nn.DataParallel(question_encoder)
    
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    
    reader = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base').to('cuda:0')
    reader = torch.nn.DataParallel(reader)
    
    reader_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
else:
    # 单GPU或CPU模式下的初始化
    ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    
    reader = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base').to(device)
    reader_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')

# 函数：编码上下文和问题
def encode_contexts(contexts):
    encoded = [ctx_encoder(**ctx_tokenizer(context[:512], return_tensors='pt').to(device)).pooler_output for context in contexts]
    return torch.cat(encoded, dim=0)

def encode_question(question):
    return question_encoder(**question_tokenizer(question[:512], return_tensors='pt').to(device)).pooler_output

# 计算问题相似度
def compute_similarity(question1, question2):
    embeddings1 = encode_question(question1).to(device)
    embeddings2 = encode_question(question2).to(device)
    similarity = cosine_similarity(embeddings1.cpu().detach().numpy(), embeddings2.cpu().detach().numpy())
    return similarity[0][0]

# 加载数据
with open('popQA_DPR.json', 'r') as file:
    data = json.load(file)

# 创建自定义数据集
dataset = CustomDataset(data)

# 创建数据加载器
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

output = []
processed_questions = set()  # 用于存储已处理的问题

# 初始化混合精度训练
scaler = GradScaler()

# 分批处理数据
for batch_data in tqdm(data_loader):
    for item in batch_data:
        question = item['question']
        
        # 检查问题是否已经处理过
        if question in processed_questions:
            continue
        
        # 将问题添加到已处理集合
        processed_questions.add(question)
        
        # 计算问题与已处理问题的相似度
        is_similar = False
        for processed_question in processed_questions:
            similarity = compute_similarity(question, processed_question)
            if similarity > 0.90:  # 设置相似度阈值为90%
                is_similar = True
                break
        
        # 如果问题与已处理问题相似，跳过该问题
        if is_similar:
            continue
        
        # 编码问题和上下文
        question_embedding = encode_question(question).to(device)
        contexts = item['negative_ctxs'] + item['hard_negative_ctxs'] + item['positive_ctxs']
        context_embeddings = encode_contexts([ctx['text'][:512] for ctx in contexts]).to(device)

        # 检索相关上下文
        scores = torch.nn.functional.cosine_similarity(question_embedding, context_embeddings).tolist()

        # 组织输出数据
        scored_contexts = []
        for ctx, score in zip(contexts, scores):
            scored_contexts.append({
                'id': ctx['id'],
                'text': ctx['text'],
                'score': score
            })

        # 阅读器提取答案
        inputs = reader_tokenizer(
            questions=[question] * len(scored_contexts),
            titles=[ctx.get('title', '') for ctx in scored_contexts],
            texts=[ctx['text'] for ctx in scored_contexts],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # 使用混合精度训练
        with autocast():
            reader_outputs = reader(**inputs)

        # 提取前5个最佳答案
        top_k = 5
        answer_start_scores = reader_outputs.start_logits
        answer_end_scores = reader_outputs.end_logits

        start_indices = torch.topk(answer_start_scores, top_k).indices[0].tolist()
        end_indices = torch.topk(answer_end_scores, top_k).indices[0].tolist()

        best_answers = []
        for start, end in zip(start_indices, end_indices):
            answer = reader_tokenizer.decode(inputs['input_ids'][0, start:end + 1])
            best_answers.append(answer)

        output.append({
            'question': question,
            'predicted_answers': best_answers,
            'ctxs': scored_contexts
        })

    # 清理GPU内存
    torch.cuda.empty_cache()

# 保存结果
with open('dpr_output.json', 'w') as f:
    json.dump(output, f, indent=4)


