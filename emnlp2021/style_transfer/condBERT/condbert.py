import torch

from collections import defaultdict


def group_by_first_token(texts, tokenizer):
    seqs = [tokenizer.encode(x, add_special_tokens=False) for x in texts]
    grouped = defaultdict(list)
    for seq in seqs:
        grouped[seq[0]].append(seq)
    return grouped


def default_chooser(hypotheses, original=None, **kwargs):
    return hypotheses[0]


class CondBertRewriter:
    def __init__(
            self,
            model,
            tokenizer,
            device,
            neg_words,
            pos_words,
            word2coef,
            token_toxicities,
            predictor=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neg_words = neg_words
        self.pos_words = pos_words
        self.word2coef = word2coef
        self.token_toxicities = token_toxicities
        self.predictor = predictor

        # calculated properties
        self.v = {v: k for k, v in tokenizer.vocab.items()}
        self.device_toxicities = torch.tensor(token_toxicities).to(self.device)

        self.neg_complex_tokens = group_by_first_token(neg_words, self.tokenizer)
        self.pos_complex_tokens = group_by_first_token(pos_words, self.tokenizer)
        self.mask_index = self.tokenizer.convert_tokens_to_ids("[MASK]")

    def toks_to_words(self, token_ids):
        """ Merge subword tokens into whole words """
        indices = []
        for i, token_id in enumerate(token_ids):
            token_text = self.v[token_id]
            if token_text.startswith('##'):
                indices.append(i)
            else:
                if indices:
                    toks = [self.v[token_ids[t]] for t in indices]
                    word = ''.join([toks[0]] + [t[2:] for t in toks[1:]])
                    yield indices, word
                indices = [i]

    def get_mask_fast(
            self,
            inp: str,
            bad_words=None,
            min_bad_score=0,
            aggressive=True,
            max_score_margin=0.5,
            label=0, # the desired label (masking tokens of the opposite label)
    ):
        if bad_words is None:
            if label == 0:
                bad_words = self.neg_complex_tokens
            else:
                bad_words = self.pos_complex_tokens

        sentences = [self.tokenizer.encode(inp, add_special_tokens=True)]
        sentences_torch = torch.tensor(sentences)
        masks = torch.zeros_like(sentences_torch)

        for sent_id, sent in enumerate(sentences):
            for first_tok_id, tok in enumerate(sent):
                for hypothesis in bad_words.get(tok, []):
                    n = len(hypothesis)
                    if sent[first_tok_id: (first_tok_id + n)] == hypothesis:
                        for step in range(n):
                            masks[sent_id, first_tok_id + step] = 1
                        # if a word has toxic prefix, it is all toxic, so we should label its suffix as well
                        for offset, next_token in enumerate(sent[(first_tok_id + n):]):
                            if self.tokenizer.convert_ids_to_tokens(next_token).startswith('##'):
                                masks[sent_id, first_tok_id + n + offset] = 1
                            else:
                                break
            if sum(masks[sent_id].numpy()) == 0 or aggressive:
                scored_words = []
                for indices, word in self.toks_to_words(sent):
                    score = self.word2coef.get(word, 0) * (1 - 2 * label)
                    if score:
                        scored_words.append([indices, word, score])
                if scored_words:
                    max_score = max(s[2] for s in scored_words)
                    if max_score > min_bad_score:
                        for indices, word, score in scored_words:
                            if score >= max(min_bad_score, max_score * max_score_margin):
                                masks[sent_id, indices] = 1

        return sentences_torch, masks

    def translate(
            self,
            ss,
            get_mask=None,
            label=0,
            prnt=True,
            raw=False,
            toxicity_penalty=15,
            contrast_penalty=0,
            mask_toxic=False,
            duplicate=False,
    ):
        if get_mask is None:
            get_mask = self.get_mask_fast
        if prnt:
            print(ss)
        if label == 0:
            input_ids, attn_mask = get_mask(ss, bad_words=self.neg_complex_tokens, label=label)
        else:
            input_ids, attn_mask = get_mask(ss, bad_words=self.pos_complex_tokens, label=label)

        if attn_mask.sum().numpy() == 0:
            return ss

        masked = torch.ones_like(input_ids) * -100
        for i in range(input_ids.shape[0]):
            masked[i][attn_mask[i] == 1] = input_ids[i][attn_mask[i] == 1]
            if duplicate:
                input_ids = torch.cat([input_ids, input_ids], axis=1)
                attn_mask = torch.cat([torch.zeros_like(attn_mask), attn_mask], axis=1)
            if mask_toxic:
                input_ids[i][attn_mask[i] == 1] = self.mask_index

        # masked = masked.to(self.device)

        input_ids = input_ids.to(self.device)

        self.model.eval()

        outputs = self.model(
            input_ids,
            token_type_ids=torch.ones_like(input_ids) * label,
        )
        if contrast_penalty:
            neg_outputs = self.model(
                input_ids,
                token_type_ids=torch.ones_like(input_ids) * (1-label),
            )
        else:
            neg_outputs = None
        if raw:
            return outputs[0]
        for i in range(input_ids.shape[0]):
            logits = outputs[-1][i][attn_mask[i] == 1]
            if toxicity_penalty:
                logits -= self.device_toxicities * toxicity_penalty * (1 - 2 * label)
            if contrast_penalty:
                neg_logits = neg_outputs[-1][i][attn_mask[i] == 1]
                scores = torch.softmax(logits, -1) - torch.softmax(neg_logits, -1) * contrast_penalty
            else:
                scores = logits
            input_ids[i][attn_mask[i] == 1] = scores.argmax(dim=1)

        result = self.tokenizer.convert_tokens_to_string(
            [self.tokenizer.convert_ids_to_tokens(i.item()) for i in input_ids[0][1:-1]]
        )
        return result.split('[SEP] [CLS] ')[-1]

    def convert_mask(self, tok_ids, mask_ids, duplicate=False, start_from=0):
        # find the first masked word, keep only its first token, get its position
        toks_tmp = [self.tokenizer.convert_ids_to_tokens(tok_ids[0])[1:-1]]
        mask_pos = None
        toks = []
        mask_toks = []
        has_mask = False
        for i, is_masked in enumerate(mask_ids[0][1:-1]):
            tok = toks_tmp[0][i]
            if not has_mask:
                if is_masked and i >= start_from and not tok.startswith('##'):
                    has_mask = True
                    mask_pos = [i]
                    mask_toks.append(tok)
                toks.append(tok)
            else:
                if not is_masked or not tok.startswith('##'):
                    toks.extend(toks_tmp[0][i:])
                    break
                else:
                    mask_toks.append(tok)
        toks = [toks]

        if duplicate:
            toks = [toks_tmp[0] + ['[SEP]'] + toks[0]]
            mask_pos[0] += len(toks_tmp[0]) + 1
        return toks, mask_pos, mask_toks

    def replacement_loop(
            self,
            text,
            span_detector=None,
            predictor=None,
            verbose=True,
            chooser=default_chooser,
            n_tokens=(1, 2, 3),
            n_top=10,
            mask_token=False,
            max_steps=1000,
            label=0,
            **predictor_args,
    ):
        if span_detector is None:
            span_detector = self.get_mask_fast
        if predictor is None:
            predictor = self.predictor
        new_text = text
        look_from = 0

        for i in range(max_steps):
            tok_ids, mask_ids = span_detector(new_text, label=label)
            if not sum(mask_ids[0][(1 + look_from):]):
                break
            toks, mask_pos, mask_toks = self.convert_mask(
                tok_ids, mask_ids, duplicate=False, start_from=look_from
            )
            if mask_pos is None:
                return new_text
            texts, scores = predictor.generate(
                toks,
                mask_pos,
                n_tokens=list(n_tokens),
                n_top=n_top,
                fix_multiunit=False,
                mask_token=mask_token,
                label=label,
                **predictor_args
            )
            old_replacement = chooser(hypotheses=texts[0], scores=scores[0], original=mask_toks)
            if isinstance(old_replacement, str):
                old_replacement = [old_replacement]
            replacement = [t for w in old_replacement for t in w.split('_')]
            if verbose:
                print(mask_toks, '->', replacement)
            new_toks = toks[0][:mask_pos[0]] + replacement + toks[0][mask_pos[0] + 1:]
            new_text = self.tokenizer.convert_tokens_to_string(new_toks)
            look_from = mask_pos[0] + len(old_replacement)
            # we could add len(replacement), but sometimes its tokens glue together like 'mental' + '##ly'
            # on the other hand, we want to skip light_##hearted fully
        return new_text
    
    def get_mlm_contrast(self, text, label=0, logs=True):
        """ Calculate difference in log likelihood (or likelihood) of a text under different token_type_ids """
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs['token_type_ids'] = inputs['token_type_ids'] * 0 + label
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            scores = self.model(**inputs, )
        pos_ll = get_ll(inputs, scores)

        inputs['token_type_ids'] = 1 - inputs['token_type_ids']
        with torch.no_grad():
            scores2 = self.model(**inputs, )
        neg_ll = get_ll(inputs, scores2)

        if logs:
            diff = pos_ll - neg_ll
        else:
            diff = torch.exp(pos_ll) - torch.exp(neg_ll)

        return diff


def get_ll(inputs, results):
    """ Calculate log likelihood of a sequence """
    probs = torch.log_softmax(results.logits, dim=-1)
    seq = inputs['input_ids']
    proba = torch.gather(probs, 2, seq[:, :, None]).squeeze(-1)
    return proba * inputs['attention_mask']