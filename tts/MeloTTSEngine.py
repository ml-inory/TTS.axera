import numpy as np
import soundfile as sf
import onnxruntime as ort
import axengine as axe
from .melo_tts.split_utils import split_sentence
from .melo_tts.text import cleaned_text_to_sequence
from .melo_tts.text.cleaner import clean_text
from .melo_tts.symbols import LANG_TO_SYMBOL_MAP
import re
import os, sys
from loguru import logger
import time

from .tts_interface import TTSInterface
from .download_utils import download_model

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def get_text_for_tts_infer(text, language_str, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)

    phone = np.array(phone, dtype=np.int32)
    tone = np.array(tone, dtype=np.int32)
    language = np.array(language, dtype=np.int32)
    word2ph = np.array(word2ph, dtype=np.int32) * 2
    word2ph[0] += 1

    return phone, tone, language, norm_text, word2ph

def split_sentences_into_pieces(text, language, quiet=False):
    texts = split_sentence(text, language_str=language)
    if not quiet:
        logger.debug(" > Text split to sentences.")
        logger.debug('\n'.join(texts))
        logger.debug(" > ===========================")
    return texts

def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    # Average pad part
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]

    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]

# 计算每个词的发音时长
def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

# 生成有overlap的slice，slice索引是对于zp的
def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices = []
    zp_slices = []
    while pn_end < len(word2pronoun):
        # 前一个slice长度大于2 且 加上现在这个字没有超过dec_len，则往前overlap两个字
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
            
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices

class TTSEngine(TTSInterface):
    def __init__(
        self,
        language: str = "ZH",
        dec_len: int = 128,
        device: str = "AX650"
    ):
        assert language in ["ZH", "EN", "JP"], "Only support ZH, EN, JP currently"
        assert device in ["AX650", "AX630C"], "Only support AX650 and AX630C currently"

        self.language = language
        if self.language == "ZH":
            self.language = "ZH_MIX_EN"
        self.dec_len = dec_len
        self.sample_rate = 44100
        self.device = device

        self.file_extension = "wav"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

        self.load_model()

    def load_model(self):
        # download model if needed
        model_path = download_model("MeloTTS")

        # copy nltk_data to home
        if not os.path.exists("~/nltk_data"):
            os.system(f"cp -rf {model_path}/nltk_data ~/")

        # load models
        if "ZH" in self.language:
            enc_model = "encoder-zh.onnx"
        else:
            enc_model = f"encoder-{self.language.lower()}.onnx"
        if "ZH" in self.language:
            dec_model = "decoder-zh.axmodel"
        else:
            dec_model = f"decoder-{self.language.lower()}.axmodel"

        self.symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[self.language])}

        self.encoder = ort.InferenceSession(os.path.join(model_path, "encoder-onnx", enc_model), providers=["CPUExecutionProvider"])
        self.decoder = axe.InferenceSession(os.path.join(model_path, "decoder-ax650" if self.device == "AX650" else "decoder-ax630c", dec_model))

        # load speaker
        self.speaker = np.fromfile(os.path.join(model_path, f"g-{self.language.lower()}.bin"), dtype=np.float32).reshape(1, 256, 1)

    def generate_audio(self, text, file_name_no_ext=None, speed=1.0):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension
        speed: float
            larger means faster

        Returns:
        str: the path to the generated audio file

        """
        file_name = self.generate_cache_file_name(
                file_name_no_ext, self.file_extension
            )

        # split text to sentences
        sens = split_sentences_into_pieces(text, self.language, quiet=True)

        # Final wav
        audio_list = []

        # Iterate over splitted sentences
        for n, se in enumerate(sens):
            if self.language in ['EN', 'ZH_MIX_EN']:
                se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)
            logger.debug(f"\nSentence[{n}]: {se}")
            # Convert sentence to phones and tones
            phones, tones, lang_ids, norm_text, word2ph = get_text_for_tts_infer(se, self.language, symbol_to_id=self.symbol_to_id)

            start = time.time()
            # Run encoder
            z_p, pronoun_lens, audio_len = self.encoder.run(None, input_feed={
                                        'phone': phones, 'g': self.speaker,
                                        'tone': tones, 'language': lang_ids, 
                                        'noise_scale': np.array([0], dtype=np.float32),
                                        'length_scale': np.array([1.0 / speed], dtype=np.float32),
                                        'noise_scale_w': np.array([0], dtype=np.float32),
                                        'sdp_ratio': np.array([0], dtype=np.float32)})
            logger.debug(f"encoder run take {1000 * (time.time() - start):.2f}ms")

            # 计算每个词的发音长度
            word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
            # 生成word2pronoun和zp的切片
            pn_slices, zp_slices = generate_slices(word2pronoun, self.dec_len)

            audio_len = audio_len[0]
            sub_audio_list = []
            for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
                zp_slice = z_p[..., zs]

                # Padding前zp的长度
                sub_dec_len = zp_slice.shape[-1]
                # Padding前输出音频的长度
                sub_audio_len = 512 * sub_dec_len

                # Padding到dec_len
                if zp_slice.shape[-1] < self.dec_len:
                    zp_slice = np.concatenate((zp_slice, np.zeros((*zp_slice.shape[:-1], self.dec_len - zp_slice.shape[-1]), dtype=np.float32)), axis=-1)

                start = time.time()
                audio = self.decoder.run(None, input_feed={"z_p": zp_slice,
                                    "g": self.speaker
                                    })[0].flatten()
                
                # 处理overlap
                audio_start = 0
                if len(sub_audio_list) > 0:
                    if pn_slices[i - 1].stop > ps.start:
                        # 去掉第一个字
                        audio_start = 512 * word2pronoun[ps.start]
        
                audio_end = sub_audio_len
                if i < len(pn_slices) - 1:
                    if ps.stop > pn_slices[i + 1].start:
                        # 去掉最后一个字
                        audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]

                audio = audio[audio_start:audio_end]
                logger.debug(f"Decode slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")
                sub_audio_list.append(audio)
            sub_audio = merge_sub_audio(sub_audio_list, 0, audio_len)
            audio_list.append(sub_audio)
        audio = audio_numpy_concat(audio_list, sr=self.sample_rate, speed=speed)
        sf.write(file_name, audio, self.sample_rate)

        return file_name