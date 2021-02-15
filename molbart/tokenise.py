import random

from pysmilesutils.tokenize import SMILESTokenizer, SMILESAtomTokenizer


class MolEncTokeniser:
    def __init__(
        self,
        smiles,
        regex=None,
        extra_tokens=None,
        begin_token="^",
        end_token="&",
        pad_token=" ",
        unk_token="?",
        mask_token="<MASK>",
        sep_token="<SEP>",
        mask_prob=0.15,
        show_mask_token_prob=0.8
    ):

        self.begin_token = begin_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.sep_token = sep_token
        self.mask_prob = mask_prob
        self.show_mask_token_prob = show_mask_token_prob

        self.unk_id = 1

        # If no regex is supplied use Atom tokeniser by default, otherwise use regex
        if regex is None:
            self._chem_tokeniser = SMILESAtomTokenizer(smiles=smiles, tokens=extra_tokens)
        else:
            self._chem_tokeniser = SMILESTokenizer(smiles=smiles, regex_tokens=[regex], tokens=extra_tokens)

        self._chem_tokeniser._unknown_id = 1

        self.vocab = None
        self.decode_vocab = None
        self.chem_token_idxs = None
        self.create_vocabulary(smiles)

        self.unk_token_cnt = {}

    def __len__(self):
        return len(self.vocab)

    def create_vocabulary(self, smiles):
        vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.begin_token: 2,
            self.end_token: 3,
            self.mask_token: 4,
            self.sep_token: 5,
        }

        chem_start_idx = len(vocab)

        for tokens in self._chem_tokeniser.tokenize(smiles):
            [vocab.setdefault(token, len(vocab)) for token in tokens]

        chem_token_idxs = range(chem_start_idx, len(vocab))
        self.chem_token_idxs = list(chem_token_idxs)

        self.vocab = vocab
        self.decode_vocab = {i: t for t, i in vocab.items()}

    def tokenise(self, sents1, sents2=None, mask=False, pad=False):
        if sents2 is not None and len(sents1) != len(sents2):
            raise ValueError("Sentence 1 batch and sentence 2 batch must have the same number of elements")

        tokens = [ts[1:-1] for ts in self._chem_tokeniser.tokenize(sents1)]
        m_tokens, token_masks = self._mask_tokens(tokens, empty_mask=not mask)

        sent_masks = None
        if sents2 is not None:
            sents2_tokens = [ts[1:-1] for ts in self._chem_tokeniser.tokenize(sents2)]
            sents2_m_tokens, sents2_masks = self._mask_tokens(sents2_tokens, empty_mask=not mask)
            tokens, sent_masks = self._concat_sentences(tokens, sents2_tokens, self.sep_token)
            m_tokens, _ = self._concat_sentences(m_tokens, sents2_m_tokens, self.sep_token)
            token_masks, _ = self._concat_sentences(token_masks, sents2_masks, False)

        tokens = [[self.begin_token] + ts + [self.end_token] for ts in tokens]
        m_tokens = [[self.begin_token] + ts + [self.end_token] for ts in m_tokens]
        token_masks = [[False] + ts + [False] for ts in token_masks]
        sent_masks = [[0] + mask + [1] for mask in sent_masks] if sent_masks is not None else None

        output = {}

        pad_masks = None
        if pad:
            tokens, pad_masks = self._pad_seqs(tokens, self.pad_token)
            m_tokens, _ = self._pad_seqs(m_tokens, self.pad_token)
            token_masks, _ = self._pad_seqs(token_masks, False)
            sent_masks, _ = self._pad_seqs(sent_masks, False) if sent_masks is not None else (None, None)
            output["pad_masks"] = pad_masks

        output["original_tokens"] = tokens

        if mask:
            output["masked_tokens"] = m_tokens
            output["token_masks"] = token_masks

        if sent_masks is not None:
            output["sentence_masks"] = sent_masks

        return output

    def _concat_sentences(self, tokens1, tokens2, sep):
        tokens = [ts1 + [sep] + ts2 for ts1, ts2 in zip(tokens1, tokens2)]
        sent_masks = [([0] * len(ts1)) + [0] + ([1] * len(ts2)) for ts1, ts2 in zip(tokens1, tokens2)]
        return tokens, sent_masks

    def detokenise(self, tokens_list):
        new_tokens_list = []
        for tokens in tokens_list:
            if tokens[0] == self.begin_token:
                tokens = tokens[1:]

            # Remove any tokens after the end token (and end token) if it's there 
            if self.end_token in tokens:
                end_token_idx = tokens.index(self.end_token)
                tokens = tokens[:end_token_idx]

            new_tokens_list.append(tokens)

        strs = ["".join(tokens) for tokens in new_tokens_list]
        return strs

    def convert_tokens_to_ids(self, token_data):
        ids_list = []
        for tokens in token_data:
            for token in tokens:
                token_id = self.vocab.get(token)
                if token_id is None:
                    self._inc_in_dict(self.unk_token_cnt, token)

            ids = [self.vocab.get(token, self.unk_id) for token in tokens]
            ids_list.append(ids)

        return ids_list

    def convert_ids_to_tokens(self, token_ids):
        tokens_list = []
        for ids in token_ids:
            for token_id in ids:
                token = self.decode_vocab.get(token_id)
                if token is None:
                    raise ValueError(f"Token id {token_id} is not recognised")
 
            tokens = [self.decode_vocab.get(token_id) for token_id in ids]
            tokens_list.append(tokens)

        return tokens_list

    def print_unknown_tokens(self):
        print(f"{'Token':<10}Count")
        for token, cnt in self.unk_token_cnt.items():
            print(f"{token:<10}{cnt}")
    
        print()

    @staticmethod
    def _inc_in_dict(coll, item):
        cnt = coll.get(item, 0)
        cnt += 1
        coll[item] = cnt

    def _mask_tokens(self, tokens, empty_mask=False):
        if empty_mask:
            mask = [[False] * len(ts) for ts in tokens]
            return tokens, mask

        mask_bools = [True, False]
        weights = [self.mask_prob, 1 - self.mask_prob]

        masked_tokens = []
        token_masks = []
        for ts in tokens:
            token_mask = random.choices(mask_bools, weights=weights, k=len(ts))
            masked = [self._mask_token(ts[i]) if m else ts[i] for i, m in enumerate(token_mask)]
            masked_tokens.append(masked)
            token_masks.append(token_mask)

        return masked_tokens, token_masks

    def _mask_token(self, token):
        rand = random.random()
        if rand < self.show_mask_token_prob:
            return self.mask_token

        elif rand < self.show_mask_token_prob + ((1 - self.show_mask_token_prob) / 2):
            token_idx = random.choice(self.chem_token_idxs)
            return self.decode_vocab[token_idx]

        else:
            return token

    @staticmethod
    def _pad_seqs(seqs, pad_token):
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks
