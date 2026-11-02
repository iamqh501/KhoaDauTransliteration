import os
import sys
import torch
import tkinter as tk
from tkinter import ttk, messagebox
import re
import random
import unicodedata
import ctypes
from difflib import SequenceMatcher

# Thêm đường dẫn để import từ các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.vocab_utils import CharTokenizer, SOS_IDX, EOS_IDX
from src.models import Seq2SeqTransformer
from evaluate import beam_search_decode

# Import Rule-based từ Module 1
from quoc_ngu_to_khoa_dau.src.rule_based import encode_safe

# --- LOAD FONT KHOA DAU (Windows) ---
FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "misc", "font", "KhoaDau-Regular_v3.otf"))

def load_custom_font(font_path):
    if os.path.exists(font_path):
        FR_PRIVATE = 0x10
        ctypes.windll.gdi32.AddFontResourceExW(font_path, FR_PRIVATE, 0)
    else:
        print(f"Cảnh báo: Không tìm thấy font tại {font_path}")

class KhoaDauToQuocNguApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Khoa Đẩu to Quốc Ngữ - AI Transliteration v2.0")
        self.root.geometry("1000x800")
        
        load_custom_font(FONT_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Tokenizer & Initialize UI components first
        self.tokenizer = CharTokenizer()
        if not self.tokenizer.load(VOCAB_PATH):
            messagebox.showerror("Lỗi", f"Không tìm thấy từ điển tại {VOCAB_PATH}")
            sys.exit(1)
            
        # Tìm danh sách các file weights (.pth) trong thư mục weights
        self.available_models = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth") and "vocab" not in f]
        if not self.available_models:
             messagebox.showwarning("Cảnh báo", "Không tìm thấy file trọng số .pth nào trong thư mục weights")
             self.available_models = ["best_model.pth"]

        self.model = None
        self.corpus = self._load_corpus()
        self.setup_ui()
        
        # Load model mặc định (chọn file đầu tiên nếu có)
        if self.available_models:
            self.load_model(self.available_models[0])

    def load_model(self, model_filename):
        model_path = os.path.join(WEIGHTS_DIR, model_filename)
        ckpt = None
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)

        if isinstance(ckpt, dict) and 'config' in ckpt:
            config = ckpt['config']
            enc_layers = config.get('num_encoder_layers', NUM_ENCODER_LAYERS)
            dec_layers = config.get('num_decoder_layers', NUM_DECODER_LAYERS)
            emb_size = config.get('emb_size', D_MODEL)
            nhead = config.get('nhead', N_HEAD)
            dim_ff = config.get('dim_feedforward', DIM_FEEDFORWARD)
        else:
            state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
            
            enc_layers = NUM_ENCODER_LAYERS
            dec_layers = NUM_DECODER_LAYERS
            emb_size = D_MODEL
            nhead = N_HEAD
            dim_ff = DIM_FEEDFORWARD
            
            if state_dict is not None:
                try:
                    # Auto-detect D_MODEL
                    if 'src_tok_emb.embedding.weight' in state_dict:
                        emb_size = state_dict['src_tok_emb.embedding.weight'].shape[1]
                    
                    # Auto-detect Layers
                    enc_keys = [k for k in state_dict.keys() if k.startswith('transformer.encoder.layers.')]
                    if enc_keys:
                        enc_layers = max([int(k.split('.')[3]) for k in enc_keys]) + 1
                        
                    dec_keys = [k for k in state_dict.keys() if k.startswith('transformer.decoder.layers.')]
                    if dec_keys:
                        dec_layers = max([int(k.split('.')[3]) for k in dec_keys]) + 1
                        
                    # Auto-detect FeedForward
                    if 'transformer.encoder.layers.0.linear1.weight' in state_dict:
                        dim_ff = state_dict['transformer.encoder.layers.0.linear1.weight'].shape[0]
                        
                    # Rule of thumb for N_HEAD based on D_MODEL in this project
                    if emb_size == 256:
                        nhead = 8
                    elif emb_size == 512:
                        nhead = 16
                except Exception as e:
                    print(f"Lỗi Auto-detect: {e}")

        self.model = Seq2SeqTransformer(
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            emb_size=emb_size,
            nhead=nhead,
            vocab_size=self.tokenizer.vocab_size,
            dim_feedforward=dim_ff,
            dropout=0.0
        ).to(self.device)

        if ckpt is not None:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"Đã tải trọng số từ {model_filename} (D_MODEL: {emb_size}, Lớp: {enc_layers})")
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Đã nạp: {model_filename} (D_MODEL: {emb_size})")
        else:
            messagebox.showwarning("Cảnh báo", f"Không thể tải: {model_path}")


    def _load_corpus(self):
        source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "quoc_ngu_to_khoa_dau", "data", "source.txt"))
        if os.path.exists(source_path):
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Tách câu đơn giản
                sentences = re.split(r'[.\n]+', content)
                return [s.strip() for s in sentences if 5 <= len(s.split()) <= 25]
        return ["Chào mừng bạn đến với hệ thống chuyển tự chữ Khoa Đẩu ."]

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)

        # --- Phía trên: Điều khiển ---
        ctrl_frame = ttk.Frame(main_frame);        ctrl_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(ctrl_frame, text="Nhập Quốc ngữ để tự động chuyển sang Khoa Đẩu:", font=("Segoe UI", 10, "italic")).pack(side="left")

        # Thêm dropdown chọn mô hình và nút Refresh
        ttk.Label(ctrl_frame, text=" AI Model:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(20, 5))
        self.model_combo = ttk.Combobox(ctrl_frame, values=self.available_models, state="readonly", width=25)
        if self.available_models:
            self.model_combo.set(self.available_models[0])
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        ttk.Button(ctrl_frame, text="🔄 Làm mới", command=self.refresh_models).pack(side="left", padx=(5, 0))

        ttk.Button(ctrl_frame, text="🎲 Câu ngẫu nhiên", command=self.random_sample).pack(side="right")
        # --- 1. Input Box (Quốc ngữ) ---
        input_label_frame = ttk.Frame(main_frame)
        input_label_frame.pack(fill="x")
        ttk.Label(input_label_frame, text="1. Văn bản Gốc (Quốc ngữ):").pack(side="left")
        ttk.Button(input_label_frame, text="⚡ Dịch", command=self.on_content_change).pack(side="right")

        self.input_text = tk.Text(main_frame, height=5, font=("Segoe UI", 13))
        self.input_text.pack(fill="x", pady=5)

        # --- 2. Middle Box (Khoa Đẩu - Hiển thị trung gian) ---
        ttk.Label(main_frame, text="2. Chữ Khoa Đẩu (Trung gian - Rule-based):").pack(anchor="w")
        self.mid_text = tk.Text(main_frame, height=5, font=("Khoa Dau Regular v3", 24), wrap="word", bg="#f9f9f9")
        self.mid_text.pack(fill="x", pady=5)
        
        # --- 3. Output Box (AI Quốc ngữ) ---
        ttk.Label(main_frame, text="3. Kết quả AI (Quốc ngữ - Highlight lỗi so với gốc):").pack(anchor="w")
        self.output_text = tk.Text(main_frame, height=8, font=("Segoe UI", 14), wrap="word")
        self.output_text.pack(fill="both", expand=True, pady=5)
        self.output_text.tag_config("error", foreground="red", underline=True)
        self.output_text.tag_config("missing", foreground="blue", font=("Segoe UI", 12, "bold"))

        # Thanh trạng thái
        self.status_var = tk.StringVar(value="Sẵn sàng")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def refresh_models(self):
        self.available_models = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth") and "vocab" not in f]
        if not self.available_models:
             self.available_models = ["best_model.pth"]
        self.model_combo['values'] = self.available_models
        
        current_val = self.model_combo.get()
        if current_val not in self.available_models:
            self.model_combo.set(self.available_models[0])
            self.on_model_changed()
        else:
            self.status_var.set("Đã làm mới danh sách mô hình.")

    def on_model_changed(self, event=None):
        selected_model = self.model_combo.get()
        self.status_var.set(f"Đang nạp mô hình: {selected_model}...")
        self.root.update_idletasks()
        self.load_model(selected_model)
        self.on_content_change()

    def preprocess_for_ai(self, text):
        """Tách dấu câu để AI xử lý đúng theo logic data_maker."""
        text = text.lower()
        # Punctuation Padding
        text = re.sub(r'([.,!?():;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def postprocess_from_ai(self, text):
        """Gộp lại dấu câu cho tự nhiên."""
        text = re.sub(r'\s+([.,!?():;])', r'\1', text)
        return text

    def postprocess_from_ai(self, text):
        """Gộp lại dấu câu cho tự nhiên theo quy tắc tiếng Việt."""
        # 1. Xóa khoảng trắng TRƯỚC các dấu ngắt: . , ! ? : ; )
        text = re.sub(r'\s+([.,!?\):;])', r'\1', text)
        # 2. Xóa khoảng trắng SAU dấu mở ngoặc: (
        text = re.sub(r'\(\s+', r'(', text)
        # 3. Đảm bảo có khoảng trắng TRƯỚC dấu mở ngoặc (ví dụ: word( -> word ()
        text = re.sub(r'(\w)\(', r'\1 (', text)
        return text

    def on_content_change(self):
        qn_input = self.input_text.get("1.0", tk.END).strip()
        if not qn_input:
            self.mid_text.delete("1.0", tk.END)
            self.output_text.delete("1.0", tk.END)
            return

        # Bước 1: Quốc ngữ -> Khoa Đẩu (Rule-based)
        kd_mid = encode_safe(qn_input)
        self.mid_text.delete("1.0", tk.END)
        self.mid_text.insert(tk.END, kd_mid)

        # Bước 2: Tách đoạn văn thành các câu để dịch riêng biệt
        sentences = re.split(r'(?<=[.?!])\s+|\n+', qn_input)
        
        full_ai_output = []
        self.status_var.set(f"AI đang dịch {len(sentences)} câu...")
        self.root.update_idletasks()
        
        try:
            for s_qn in sentences:
                if not s_qn.strip(): continue
                
                # Chuyển từng câu sang Khoa Đẩu trung gian
                s_kd = encode_safe(s_qn)
                
                # Tiền xử lý (Padding dấu câu cho đúng chuẩn data huấn luyện)
                ai_input = self.preprocess_for_ai(s_kd)
                
                # Thêm dấu chấm giả nếu thiếu
                has_end_punc = re.search(r'[.?!]$', ai_input)
                if not has_end_punc:
                    ai_input += " ."
                
                src_ids = self.tokenizer.encode(ai_input)
                
                # Cắt cụt input nếu dài hơn MAX_LEN để tránh crash GPU
                if len(src_ids) > MAX_LEN:
                    src_ids = src_ids[:MAX_LEN-1] + [src_ids[-1]]
                
                src_tensor = torch.tensor(src_ids).unsqueeze(1).to(self.device)
                
                # Đối với nn.Transformer, nếu src_mask không che gì cả (all False), tốt nhất truyền None 
                src_mask = None
                
                with torch.no_grad():
                    out_ids = beam_search_decode(
                        self.model, src_tensor, src_mask, 
                        max_len=MAX_LEN, 
                        start_symbol=SOS_IDX, 
                        end_symbol=EOS_IDX, 
                        device=self.device, 
                        beam_width=3
                    )
                
                s_output_raw = self.tokenizer.decode(out_ids)
                s_output = self.postprocess_from_ai(s_output_raw)
                
                # Xóa dấu chấm giả
                if not has_end_punc and s_output.endswith("."):
                    s_output = s_output[:-1].strip()
                
                full_ai_output.append(s_output)
            
            # Ghép các câu lại
            final_output = " ".join(full_ai_output)
            
            # Bước 4: Hiển thị so sánh (Highlight lỗi thực tế của AI)
            self.display_comparison(qn_input, final_output)
            self.status_var.set(f"Hoàn tất ({len(sentences)} câu)")
            
        except Exception as e:
            self.status_var.set(f"Lỗi: {str(e)}")
            import traceback
            traceback.print_exc()

    def display_comparison(self, original, predicted):
        self.output_text.delete("1.0", tk.END)
        
        orig_tokens = re.findall(r'\w+|[^\w\s]|\s+', original.lower())
        pred_tokens = re.findall(r'\w+|[^\w\s]|\s+', predicted.lower())
        
        orig_words = [t for t in orig_tokens if re.match(r'\w+', t)]
        pred_words = [t for t in pred_tokens if re.match(r'\w+', t)]
        
        sm = SequenceMatcher(None, orig_words, pred_words)
        
        # word_status: index trong pred_words -> tag ("" hoặc "error")
        word_status = {}
        # missing_markers: index trong pred_words -> danh sách các cụm từ bị thiếu TRƯỚC từ đó
        missing_markers = {i: [] for i in range(len(pred_words) + 1)}
        
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for j in range(j1, j2):
                    word_status[j] = ""
            elif tag == 'replace':
                for j in range(j1, j2):
                    word_status[j] = "error"
                missing_markers[j1].append(f"[ {' '.join(orig_words[i1:i2])} ]")
            elif tag == 'insert':
                for j in range(j1, j2):
                    word_status[j] = "error"
            elif tag == 'delete':
                missing_markers[j1].append(f"[ {' '.join(orig_words[i1:i2])} ]")

        # Duyệt qua pred_tokens để hiển thị lên UI
        curr_word_idx = 0
        for token in pred_tokens:
            if re.match(r'\w+', token):
                # Chèn marker thiếu TRƯỚC từ hiện tại nếu có
                for marker in missing_markers[curr_word_idx]:
                    self.output_text.insert(tk.END, marker + " ", "missing")
                
                status = word_status.get(curr_word_idx, "error")
                self.output_text.insert(tk.END, token, status)
                curr_word_idx += 1
            else:
                self.output_text.insert(tk.END, token)
        
        # Chèn marker thiếu ở cuối văn bản nếu AI dịch thiếu phần cuối
        for marker in missing_markers[curr_word_idx]:
            self.output_text.insert(tk.END, " " + marker, "missing")


    def random_sample(self):
        if self.corpus:
            sample = random.choice(self.corpus)
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, sample)
            self.on_content_change()

if __name__ == "__main__":
    root = tk.Tk()
    # Giảm kích thước cửa sổ một chút để vừa màn hình
    root.geometry("1000x700")
    app = KhoaDauToQuocNguApp(root)
    root.mainloop()
