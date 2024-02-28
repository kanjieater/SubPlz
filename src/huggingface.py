import whisper
from whisper.audio import load_audio, log_mel_spectrogram, N_SAMPLES, N_FRAMES
from whisper import decoding
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer
import torch
import numba
import numpy as np
from typing import Optional, Union, Tuple
from types import MethodType

class DecodingTask(decoding.DecodingTask):
    def _main_loop(self, audio_features, tokens):
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        prev_logits = torch.tensor([], device=audio_features.device)

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)
                prev_logits = torch.concat([prev_logits, logits], dim=-2)

                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs, prev_logits

    def run(self, mel):
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                decoding.DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0) # Think this is a bug in the original impl?

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs, logits = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        logits = logits[[i * self.n_group + j for i, j in enumerate(selected)]]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            decoding.DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=whisper.utils.compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ], logits

@torch.no_grad()
def decode(model, mel, options, **kwargs):
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result, logits = DecodingTask(model, options).run(mel)
    return result[0] if single else result, logits

def similarity(l1, l2):
    sm = torch.zeros([*l1.shape[:-2], l1.shape[-2], l2.shape[-2]])
    for i in range(l1.shape[-2]): # sm = (l1 * l2).sum(-1) # The dream
        m = l1[:, [i]] * l2
        sm[..., i, :] = -2 * (1 - m.sqrt().sum(-1)).sqrt() + 1
    return sm

def traceback(c, mi, mj, fl, fs, sl, ss):
    ot = []
    t1, t2 = [], []
    def score(x):
        return sum(x)/((5 + len(x))/6)
    def push():
        nonlocal t1, t2
        if not len(t1) and not len(t2): return
        s1, s2 = [fl[mi+k][t] for k, t in enumerate(reversed(t1))], [sl[mj+k][t] for k, t in enumerate(reversed(t2))]
        ot.extend(t1 if score(s1) > score(s2) else t2)
        t1, t2 = [], []
    while mi > 0 and mj > 0:
        f = c[[mi-1, mi, mi-1], [mj-1, mj-1, mj]]
        m = f.argmax()
        if f[m] == 0: break
        if m == 0:
            push()
            t1.append(fs[mi-1])
            t2.append(ss[mj-1])
            mi, mj = mi-1, mj-1
        elif m == 1:
            t2.append(ss[mj-1])
            mj = mj-1
        else:
            t1.append(fs[mi-1])
            mi = mi-1
    push()
    return ot[::-1], mi, mj

@numba.jit(nopython=True, parallel=True)
def align(sm: np.ndarray, gap=-1):
    N, M = sm.shape[-2:]
    cost = np.zeros((N+1, M+1), dtype=np.float32)
    m, mi, mj = 0, 0, 0
    for i in range(1, N+1):
        for j in range(1, M+1):
            c0 = cost[i - 1, j - 1] + sm[i-1, j-1]
            c1 = cost[i - 1, j] + gap
            c2 = cost[i, j - 1] + gap
            c = max(c1, c2, c0.item(), 0.0)
            cost[i, j] = c
            if c > m:
                m, mi, mj = c, i, j
    return cost, mi, mj

# Only half done
def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    batches: int = 1,
    overlap: int = 10,
    **decode_options,
):
    if initial_prompt is not None: # Temperature and compression ratio are also ignored for now
        raise Exception("Initial_prompt is not supported")
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = audio.load_audio(audio)
        audio = torch.from_numpy(audio)
    audio = audio.to(model.device).to(dtype)

    # Pad 30-seconds of silence to the input audio, for slicing
    # mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    # content_frames = mel.shape[-1] - N_FRAMES

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
                )
            mel_segment = log_mel_spectrogram(audio[:, N_SAMPLES], model.dims.n_mel)
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    left = 30 - overlap
    last = torch.zeros((1, 0, model.dims.n_vocab))
    last_tokens = DecodingResult(audio_features=None, language=decode_options['language'])
    for i in range(0, audio.shape[0], left * 16000 * batches):
        x = audio[i:i+left * 16000 * batches + overlap * 16000]
        mel = log_mel_spectrogram(x)
        mels = []
        for k in range(batches):
            chunk = mel[:, k * left*100: k * left*100 + 3000]
            if chunk.shape[-1] == 0: break
            if chunk.shape[-1] < 3000: chunk = audio.pad_or_trim(chunk, audio.N_FRAMES)
            mels.append(chunk.unsqueeze(0))
        mels = torch.concat(mels, dim=0)
        mels = mels.half() if decode_options['fp16'] else mels
        audio_features = model.encoder(mels)
        result, logits = model.decode(audio_features, DecodingOptions(**decode_options)) # TODO: options
        for i in result:
            print(tokenizer.decode_with_timestamps(i.tokens))

        ls = logits.shape[1]
        for i in range(logits.shape[0]):
            if i == 0:
                fl, fs = last, np.array(last_tokens.tokens)
            else:
                fl, fs = logits[i-1], np.array(result[i-1].tokens)
            sl, ss = logits[i].clone(), np.array(result[i].tokens)
            fl, sl = fl[3: 3+len(fs)].log_softmax(-1), sl[3: 3+len(ss)].log_softmax(-1)

            # Edit the timestamps of sl relative to fl
            if len(ss) > sl.shape[0]: # What? Feels like a bug
                ss = ss[:sl.shape[0]-len(ss)]
            timestamps = ss >= tokenizer.timestamp_begin
            overlap_timestamps = int(tokenizer.timestamp_begin + overlap // 0.02)+1
            left_timestamps = int(tokenizer.timestamp_begin + left // 0.02)+1
            sl[timestamps, left_timestamps: int(tokenizer.timestamp_begin + 30//0.02+1)] = sl[timestamps, tokenizer.timestamp_begin:overlap_timestamps]
            sl[timestamps, tokenizer.timestamp_begin:overlap_timestamps]  = -np.inf#sl[sl.ge(tokenizer.timestamp_begin): tokenizer.timestamp_begin + left / 0.02: tokenizer.timestamp_begin + 30/0.02] =
            sm = similarity(fl.unsqueeze(0).exp(), sl.unsqueeze(0).exp())[0].numpy()

            c, mi, mj = align(sm)
            shared, ni, nj = traceback(c, mi, mj, fl, fs, sl, ss)

            print("Shared:", tokenizer.decode_with_timestamps(shared))
        last = logits[-1]
        last_tokens = result[-1]

def modify_model(model):
    model.decode = MethodType(decode, model)
    model.transcribe = MethodType(transcribe, model)
    model.huggingface = True
    return model
