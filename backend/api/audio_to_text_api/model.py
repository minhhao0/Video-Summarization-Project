
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import subprocess
from pydub import AudioSegment
import os
class Mp4_to_text:
    def __init__(self,model_id="D:\\Phowisper", language = "vi", time_chunk_mp3=30 * 1000, folder_path_chunk = "../../audio_chunk/filemp3_chunk", output_file_path_mp3 = "../../audio/output3.mp3",
                 output_file_path_txt = "../../text/output.txt", gpu=False):
        self.model=AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        self.model.config.tie_word_embeddings = False
        self.processor = AutoProcessor.from_pretrained(model_id)
        print('load model successfully')
        # ngôn ngữ của video đầu vào hiện tại chỉ lấy tiếng việt. Trong tương lai sẽ update và thêm các model để xử lý video tiếng nước ngoài
        self.language = language
        # đường dẫn của video mp4 đầu vào
        # độ dài mỗi chunk để model trích xuất khuyến nghị không nên sửa để tránh model bị quá tham số -> lỗi
        self.chunk_length_ms = time_chunk_mp3
        # đường dẫn của folder tổng lưu trữ chunk
        self.base_folder = folder_path_chunk
        # đường dẫn của file audio mp3 sau khi được chuyển từ video mp4
        self.op_file_mp3 = output_file_path_mp3
        # đường dẫn kết quả trả về file txt
        self.op_file_txt = output_file_path_txt
        # nếu máy có gpu thì là True không thì là False tối ưu cho tốc độ nếu có gpu
        self.gpu = gpu

        # ví dụ:
        # self.language = "vi"
        # self.ip_path = "/kaggle/input/datasets/fasgadhsxnzmjj/mp4-dataset/treemp4.mp4"
        # self.chunk_length_ms = 30 * 1000
        # self.base_folder = "/kaggle/working/filemp3_chunk"
        # self.op_file_mp3 = "/kaggle/working/output2.mp3"
        # self.gpu = "True"

    def split_audio_into_chunks(self, file_path):
        folder = os.path.join(
            self.base_folder,
            os.path.splitext(os.path.basename(file_path))[0]
        )

        os.makedirs(folder, exist_ok=True)

        audio = AudioSegment.from_file(file_path)
        duration = len(audio)

        for i, start in enumerate(range(0, duration, self.chunk_length_ms)):
            chunk = audio[start:start + self.chunk_length_ms]
            chunk_path = os.path.join(folder, f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")

        return folder

    def mp4_to_mp3(self,ip_path):
        command = [
            "ffmpeg", "-i",ip_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ab", "192k",
            self.op_file_mp3
        ]
        subprocess.run(command, check=True)

    def mp3_to_text_phoWhisper(self, file_mp3):
        audio, sr = librosa.load(file_mp3, sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        with torch.no_grad():
            ids = self.model.generate(inputs["input_features"])

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def batch_mp3_to_text(self, file_list):
        # dùng nếu máy có GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        audios = []

        # Load toàn bộ audio
        for file_mp3 in file_list:
            audio, sr = librosa.load(file_mp3, sr=16000)
            audios.append(audio)

        # Convert batch
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True  # QUAN TRỌNG
        )

        # Đưa lên GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference 1 lần
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_new_tokens=256,
                do_sample=False
            )

        # Decode batch
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return texts

    def save_to_txt(self, text, output_path="output.txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    def op_vi(self,input_file_path_mp4):
        result = []

        self.mp4_to_mp3(input_file_path_mp4)
        folder = self.split_audio_into_chunks(self.op_file_mp3)

        files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".mp3")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        if not self.gpu:
            for f in files:
                path = os.path.join(folder, f)
                text = self.mp3_to_text_phoWhisper(path)
                result.append(text)

        if self.gpu:
            file_paths = [os.path.join(folder, f) for f in files]
            # chia batch (rất quan trọng nếu nhiều file)
            batch_size = 4

            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]
                texts = self.batch_mp3_to_text(batch_files)
                result.extend(texts)

        # return result
        full_text = " ".join(result)
        self.save_to_txt(full_text, self.op_file_txt)
        for f in files:
            print(f)
            if os.path.isfile(f):
                os.remove(f)
            else:
                continue
        return full_text


# ngôn ngữ của video đầu vào hiện tại chỉ lấy tiếng việt. Trong tương lai sẽ update và thêm các model để xử lý video tiếng nước ngoài

# đường dẫn của video mp4 đầu vào

# độ dài mỗi chunk để model trích xuất khuyến nghị không nên sửa để tránh model bị quá tham số -> lỗi
# đường dẫn của folder tổng lưu trữ chunk

# đường dẫn của file audio mp3 sau khi được chuyển từ video mp4

# đường dẫn kết quả trả về file txt

# nếu máy có gpu thì là True không thì là False tối ưu cho tốc độ nếu có gpu

