import argparse
from RetrieverPipeline import RetrieverPipeline

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grammar', default='/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/data/GrammarBook_structured.json', help='GrammarBook JSON')
    p.add_argument('--qa', default='/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/getPerplexity/result/lowest_ppl_results(kanana_basic_top5).json', help='lowest_ppl_results JSON')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_n', type=int, default=30)
    p.add_argument('--out', default='dual_rerank_results.json')
    p.add_argument('--biencoder_name', default='jhgan/ko-sroberta-multitask')
    p.add_argument('--cross_name', default='snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    pipe = RetrieverPipeline(
        grammar_path=args.grammar,
        qa_path=args.qa,
        biencoder_name=args.biencoder_name,
        cross_name=args.cross_name,
        device=args.device
    )
    pipe.run(
        top_k=args.top_k, 
        top_n=args.top_n, 
        save_path=args.out
    )