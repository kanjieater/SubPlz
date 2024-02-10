import torch
from torch import Tensor
import torch.nn.functional as F

from whisper.decoding import DecodingOptions, DecodingResult, DecodingTask, TokenDecoder
from whisper.audio import N_FRAMES
from whisper.tokenizer import get_tokenizer

from typing import TYPE_CHECKING, List, Optional, Union

class DirectedDecoder(TokenDecoder):
    def __init__(self, segments, tokenizer):
        self.tokenizer = tokenizer
        self.text_segments = segments
        self.token_segments = [self.tokenizer.encode(i) for i in self.text_segments]
        # self.token_tensor = torch.tensor(self.token_segments)

        self.cseg = 0
        self.cpos = 0
        self.oseg = 0
        self.opos = 0

        self.timestamps = None
        self.tiktok = False
        self.ntimestamps = 0
        self.seek = None

        self.end = False

    def preset():
        self.cseg = self.oseg
        self.cpos = self.opos

    def padvance(): # Make this a class or something
        self.oseg = self.cseg
        self.opos = self.cpos

    def to_ts(self, timestamp):
        t = int(((timestamp - self.seek) / N_FRAMES * 30) // 0.02)
        t = min(30 // 0.02, max(0, t))
        return self.tokenizer.timestamp_begin + t

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor):
        assert tokens.shape[0] == 1, "batch size != 1"

        # print(self.tokenizer.decode_with_timestamps(tokens[0].tolist()))

        mout, tout = logits.argmax(dim=-1), torch.tensor([self.token_segments[self.cseg][self.cpos]], device=logits.device)
        next_tokens = mout if mout >= self.tokenizer.eot or self.end else tout
        if next_tokens == tout:
            self.cpos += 1
            if self.cpos == len(self.token_segments[self.cseg]):
                self.cseg += 1
                self.cpos = 0
            if self.cseg == len(self.token_segments):
                self.cseg, self.cpos = -1, -1
                self.end = True

        if mout >= self.tokenizer.timestamp_begin:
            # print(self.tokenizer.decode_with_timestamps(mout.tolist()))
            start, end = self.to_ts(self.timestamps[self.ntimestamps]), self.to_ts(self.timestamps[self.ntimestamps+1])
            # print(self.tokenizer.decode_with_timestamps([start, end]))
            while self.ntimestamps < len(self.timestamps)-2 and mout > end:
                self.ntimestamps += 2
                start, end = self.to_ts(self.timestamps[self.ntimestamps]), self.to_ts(self.timestamps[self.ntimestamps+1])
                # print(self.tokenizer.decode_with_timestamps([start, end]))
            # print(self.tiktok)
            if mout < start and (not self.tiktok or self.ntimestamps == 0):
                mout = torch.tensor([start], device=logits.device)
            elif mout < start and self.tiktok:
                mout = torch.tensor([self.to_ts(self.timestamps[self.ntimestamps-1])], device=logits.device)

            self.tiktok = not self.tiktok

            next_tokens = mout
        # print(next_tokens)

        current_logprobs = F.log_softmax(logits.float(), dim=-1)[torch.arange(tokens.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.tokenizer.eot)

        next_tokens[tokens[:, -1] == self.tokenizer.eot] = self.tokenizer.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        return tokens, (tokens[:, -1] == self.tokenizer.eot).all()

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        tokens = F.pad(tokens, (0, 1), value=self.tokenizer.eot)
        return tokens, sum_logprobs.tolist()

def decode(
    model: "Whisper",
    mel: Tensor,
    decoder: TokenDecoder,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    task = DecodingTask(model, options)
    task.decoder = decoder
    result = task.run(mel)
    return result[0] if single else result
