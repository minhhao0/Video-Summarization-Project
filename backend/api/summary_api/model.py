import re
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 150
MIN_NEW_TOKENS = 60
CHUNK_SIZE = 250
class Summarizer:
    def __init__(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Load summary model successfully !!!")
        self.model.eval()

    def clean_subtitle_text(self,text: str) -> str:
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            if not line:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            if re.search(r"\d{2}:\d{2}:\d{2}", line):
                continue
            if line.upper() == "WEBVTT":
                continue

            cleaned_lines.append(line)

        text = " ".join(cleaned_lines)
        text = re.sub(r"\[.*?\]", " ", text)
        text = re.sub(r"\(.*?\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def read_text_file(self,file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if ext == ".txt":
            return content

        if ext in [".srt", ".vtt"]:
            return self.clean_subtitle_text(content)

        return content

    def normalize_text(self,text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def split_sentences(self,text: str):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_text_into_chunks(self,text: str, max_tokens: int = CHUNK_SIZE) -> List[str]:
        sentences = self.split_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            temp = current_chunk + " " + sentence

            if len(temp) < 1000:
                token_count = len(self.tokenizer.encode(temp, add_special_tokens=False))
            else:
                token_count = max_tokens + 1

            if token_count <= max_tokens:
                current_chunk = temp
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    def summarize_chunk(self,text: str) -> str:
        text = self.normalize_text(text)

        inputs = self.tokenizer(
            text,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                num_beams=3,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary.strip()

    # 9. TÓM TẮT VĂN BẢN DÀI
    def summarize_long_text(self,text: str) -> str:
        text = self.normalize_text(text)

        token_len = len(self.tokenizer.encode(text, add_special_tokens=False))

        # nếu ngắn → tóm tắt luôn
        if token_len <= MAX_INPUT_TOKENS:
            return self.summarize_chunk(text)

        # chia đoạn
        chunks = self.split_text_into_chunks(text)
        print(f"Số đoạn: {len(chunks)}")

        chunk_summaries = []

        for i, chunk in enumerate(chunks, 1):
            print(f"Đang xử lý đoạn {i}/{len(chunks)}...")
            try:
                summary = self.summarize_chunk(chunk)
                if summary:
                    chunk_summaries.append(summary)
            except Exception as e:
                print(f"Lỗi đoạn {i}: {e}")

        #  GHÉP LẠI
        final_summary = " ".join(chunk_summaries)

        # clean format
        final_summary = final_summary.replace(" .", ".")
        final_summary = final_summary.replace(" ,", ",")

        return final_summary.strip()
    def run(self,file_path):
        input_file = file_path

        if not os.path.exists(input_file):
            print("Không tìm thấy file.")
            return

        text = self.read_text_file(input_file)

        if not text.strip():
            print("File không có nội dung hợp lệ.")
            exit()

        summary = self.summarize_long_text(text)
        return summary