"""
The models should be downloaded from
https://drive.google.com/drive/folders/1lBN2nbzxtpqbPUyeURtzt0k1kBY6u6Mj
The source is http://style.cs.umass.edu
"""

import torch
import numpy as np
from .sim_models import WordAveraging
from .sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm


class SimilarityEvaluator:
    def __init__(
        self,
        model_path='models/sim/sim.pt',
        tokenizer_path='models/sim/sim.sp.30k.model',
        gpu=False
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tok = TreebankWordTokenizer()
        kw = {}
        if not torch.cuda.is_available():
            kw['map_location'] = torch.device('cpu')
        model = torch.load(self.model_path, **kw)
        state_dict = model['state_dict']
        vocab_words = model['vocab_words']
        args = model['args']
        if gpu is False:
            args.gpu = -1
        # turn off gpu
        self.model = WordAveraging(args, vocab_words)
        self.model.load_state_dict(state_dict, strict=True)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.tokenizer_path)
        self.model.eval()

    def make_example(self, sentence):
        sentence = sentence.lower()
        sentence = " ".join(self.tok.tokenize(sentence))
        sentence = self.sp.EncodeAsPieces(sentence)
        wp1 = Example(" ".join(sentence))
        wp1.populate_embeddings(self.model.vocab)
        return wp1

    def find_similarity(self, s1, s2):
        with torch.no_grad():
            s1 = [self.make_example(x) for x in s1]
            s2 = [self.make_example(x) for x in s2]
            wx1, wl1, wm1 = self.model.torchify_batch(s1)
            wx2, wl2, wm2 = self.model.torchify_batch(s2)
            scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            return [x.item() for x in scores]

    def find_similarity_batched(self, inputs, preds, batch_size=32):
        assert len(inputs) == len(preds)
        sim_scores = []
        for i in range(0, len(inputs), batch_size):
            sim_scores.extend(
                self.find_similarity(inputs[i:i + batch_size], preds[i:i + batch_size])
            )
        return np.array(sim_scores)

    def embed_texts(self, texts, batch_size=128):
        result = []
        for i in range(0, len(texts), batch_size):
            wx, wl, wm = self.model.torchify_batch([self.make_example(x) for x in texts[i:i+batch_size]])
            with torch.no_grad():
                tensors = torch.nn.functional.normalize(self.model.encode(wx, wm, wl))
            result.append(tensors.cpu().numpy())
        return np.concatenate(result)
