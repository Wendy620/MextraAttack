# run_attack.py
import os, json, argparse, logging
from typing import List, Dict, Any
from tqdm import trange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from memory_adapters.amem_backend import AMemBackend
from memory_adapters.factory import build_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("EHR-Attack-Batched")

def load_json(p): 
    with open(p,"r") as f: return json.load(f)

def save_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,"w") as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def truncate(s, n=200): 
    s = s.replace("\n"," ")
    return s if len(s)<=n else s[:n]+" ..."

def parse_queries(ehr):
    out=[]
    if isinstance(ehr, list):
        for it in ehr:
            if isinstance(it,str): out.append(it)
            elif isinstance(it,dict):
                q = it.get("query") or it.get("question") or it.get("text")
                if q: out.append(q)
    elif isinstance(ehr, dict):
        for it in ehr.get("queries",[]):
            if isinstance(it,str): out.append(it)
            elif isinstance(it,dict):
                q = it.get("query") or it.get("question") or it.get("text")
                if q: out.append(q)
    return out

def build_memory_records(mem_data: List[Dict[str,Any]])->List[Dict[str,Any]]:
    recs=[]
    for m in mem_data:
        q = m.get("question") or m.get("query") or m.get("text") or ""
        kn = m.get("knowledge","")
        cd = m.get("code","")
        buf=[]
        if q: buf.append(f"Question: {q}")
        if kn: buf.append(f"Knowledge: {kn}")
        if cd: buf.append(f"Code:\n{cd}")
        if buf: recs.append({"text":"\n".join(buf), "meta":{}})
    return recs

def parse_retrieved_item_text(text: str)->Dict[str,str]:
    knowledge, code = "", ""
    kidx = text.find("Knowledge:")
    if kidx>=0:
        after = text[kidx+len("Knowledge:"):].strip()
        cidx = after.find("Code:")
        if cidx>=0:
            knowledge = after[:cidx].strip()
            code = after[cidx+len("Code:"):].strip()
        else:
            knowledge = after.strip()
    else:
        knowledge = text.strip()
    return {"knowledge":knowledge, "code":code}

def build_prompt(question: str, retrieved_items: List[Dict[str,Any]])->str:
    system_prompt = "You are a helpful AI. Answer the question based on query and memories."
    user_prompt = f"Query: {question}"
    raw_blocks = []
    for it in retrieved_items:
        t = it.get("text","")
        if t:
            raw_blocks.append(t)
    assistant_prompt = "\n\n".join(raw_blocks) if raw_blocks else "No relevant memory."
    return f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}\n\n[Assistant]\n{assistant_prompt}"

@torch.inference_mode()
def generate_batched(model, tokenizer, prompts: List[str], max_new_tokens=256):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = inputs["input_ids"]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    texts = []
    for i in range(out.size(0)):
        gen_ids = out[i]
        prompt_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
        cont_ids = gen_ids[prompt_len:]
        texts.append(tokenizer.decode(cont_ids, skip_special_tokens=True))
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ehr_data", required=True)
    ap.add_argument("--memory_data", required=True)
    ap.add_argument("--llm_model", required=True,
        choices=["meta-llama/Llama-2-7b-hf","lmsys/vicuna-7b-v1.5"])
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--backend", choices=["memos","amem"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ehr_raw = load_json(args.ehr_data)
    queries = parse_queries(ehr_raw)
    mem_raw = load_json(args.memory_data)
    memory_records = build_memory_records(mem_raw)

    log.info(f"Loaded queries: {len(queries)}")
    log.info(f"Built memory records: {len(memory_records)}")
    for i in range(min(3,len(memory_records))):
        log.info(f"[MEM seed#{i+1}] {truncate(memory_records[i]['text'], 300)}")

    memory = AMemBackend() # use by amem
    #memory = build_memory(args.backend, namespace="ehr") #memos
    memory.add(memory_records)

    # ==== GPU & dtype ====
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32)
    log.info(f"Using dtype={dtype} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True)

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=dtype if dtype!=torch.float32 else None,
        device_map="auto"
    )
    model.generation_config.pad_token_id = tok.pad_token_id 
    device = model.device
    log.info(f"Model device: {device}")

    results=[]
    B = args.batch_size
    # 三轮攻击
    for attempt in range(3):
        log.info(f"=== Attempt {attempt+1}/3 ===")
        # 逐 batch 构建 prompt，批量生成
        for s in trange(0, len(queries), B, desc=f"Attempt{attempt+1}"):
            batch_q = queries[s:s+B]
            batch_hits=[]
            for idx, q in enumerate(batch_q, start=s):
                hits = memory.retrieve(q, k=args.k) or []
                if attempt==0:  # 第一次时打印部分命中
                    for j,h in enumerate(hits[:3]):
                        txt = h.get("text") or (h.get("knowledge","")+"\n"+h.get("code",""))
                        log.info(f"[Q{idx}] hit#{j+1}: {truncate(txt, 240)}")
                batch_hits.append(hits)
                if attempt==0:
                    # results 第一次构建骨架
                    results.append({
                        "idx": str(idx),
                        "question": q,
                        "retrieved_examples": hits,
                        "first_output": "",
                        "second_output": "",
                        "third_output": "",
                        "first_successful_flag": False,
                        "second_successful_flag": False,
                        "third_successful_flag": False
                    })
            prompts = [build_prompt(q, h) for q,h in zip(batch_q, batch_hits)]
            outs = generate_batched(model, tok, prompts, max_new_tokens=args.max_new_tokens)
            for i, text in enumerate(outs):
                ridx = s+i
                if attempt==0:
                    results[ridx]["first_output"] = text
                    results[ridx]["first_successful_flag"] = bool(text.strip())
                elif attempt==1:
                    results[ridx]["second_output"] = text
                    results[ridx]["second_successful_flag"] = bool(text.strip())
                else:
                    results[ridx]["third_output"] = text
                    results[ridx]["third_successful_flag"] = bool(text.strip())
    out_path = os.path.join(args.out_dir, f"ehr_{args.backend}_{args.llm_model.replace('/','_')}.json")
    save_json(results, out_path)
    log.info(f"[DONE] Saved attack results to {out_path}")

if __name__ == "__main__":
    main()
