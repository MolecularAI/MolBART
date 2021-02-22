import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LambdaLR
from onmt.modules import MultiHeadedAttention
from functools import partial


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Abstract Models ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _AbsTransformerModel(pl.LightningModule):
    def __init__(
        self, 
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        dropout=0.1,
        max_seq_len=None,
        batch_size=None,
        epochs=None,
        mask_prob=None,
        augment=None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.dropout = dropout

        # Additional hparams passed in are saved
        self.max_seq_len = max_seq_len if max_seq_len is not None else "Unknown"
        self.batch_size = batch_size if batch_size is not None else "Unknown"
        self.epochs = epochs if epochs is not None else "Unknown"
        self.mask_prob = mask_prob if mask_prob is not None else "Unknown"
        self.augment = augment if augment is not None else "Unknown"

        self.save_hyperparameters()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "masked_tokens": tensor of token_ids of shape (seq_len, batch_size),
                "pad_masks": bool tensor of padded elems of shape (seq_len, batch_size),
                "sentence_masks" (optional): long tensor (0 or 1) of shape (seq_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """

        raise NotImplementedError()

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor)
        """

        raise NotImplementedError()

    def _log_validation_metrics(self, batch_input, model_output):
        """ Calculate validation metrics for the model and log

        Args:
            batch input (dict): Input given to model
            model_output (dict): Output from model
        """

        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("train_loss", loss, on_step=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("val_loss", loss, on_epoch=True, on_step=False, logger=True)
        self._log_validation_metrics(batch, model_output)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.num_steps)
        lr_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optim], [lr_dict]

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz):
        """ 
        Method from Pytorch transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val)


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Pre-train Models --------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        dropout=0.1,
        max_seq_len=None,
        batch_size=None,
        epochs=None,
        mask_prob=None,
        augment=None
    ):
        super().__init__(
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            dropout=dropout,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            epochs=epochs,
            mask_prob=mask_prob,
            augment=augment
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.num_beams = 5

        enc_norm = nn.LayerNorm(d_model)
        dec_norm = nn.LayerNorm(d_model)

        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_mask_fc = nn.Linear(d_model, vocab_size)
        self.token_mask_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """

        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(self.device)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output = self.decoder(
            decoder_embs,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask.clone()
        )

        token_output = self.token_mask_fc(model_output)

        output = {
            "model_output": model_output,
            "token_output": token_output
        }

        return output

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(self.device)

        model_output = self.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_mask_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_char_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)

        # self.log("val_loss", loss.item(), on_epoch=True, on_step=False)
        # self.log("val_char_acc", char_acc, on_epoch=True, on_step=False)

        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        val_outputs = {
            "val_loss": loss.item(),
            "val_token_acc": token_acc,
            "val_perplexity": perplexity,
            "val_molecular_accuracy": metrics["accuracy"],
            "val_invalid_smiles": metrics["invalid"]
        }

        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_pad_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.token_mask_loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        # Mask out loss from padded tokens, normalise by length of each sequence
        # loss = loss * ~target_mask
        # loss = loss.sum(dim=0) / (~target_mask).sum(dim=0)
        # loss = loss.mean()

        num_tokens = (~target_mask).sum()
        loss = loss.sum() / num_tokens

        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_pad_mask = batch_input["target_pad_mask"]
        vocab_dist_output = model_output["token_output"]

        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * ~target_pad_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = (~target_pad_mask).sum(dim=0)
        exp = - (1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean().item()

    def _calc_char_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        token_output = model_output["token_output"]

        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)

        num_correct = correct_ids.sum().cpu().detach().item()
        total = correct_ids.numel()

        accuracy = num_correct / total
        return accuracy

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        encode_input = {
            "encoder_input": enc_input,
            "encoder_pad_mask": enc_mask
        }
        memory = self.encode(encode_input)
        mem_mask = enc_mask.clone()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)
        self.sampler.device = self.device

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        model_output = self.decode(decode_input)
        return model_output


# ----------------------------------------------------------------------------------------------------------
# ------------------------------------------- Fine-tune Models ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class ReactionPredModel(pl.LightningModule):
    def __init__(
        self, 
        model,
        decode_sampler,
        lr,
        weight_decay,
        num_steps,
        epochs,
        schedule,
        swa_lr,
        pad_token_idx,
        max_seq_len=None,
        batch_size=None,
        clip_grad=None,
        augment=None,
        warm_up_steps=None,
        acc_batches=None
    ):
        super(ReactionPredModel, self).__init__()

        self.model = model
        self.sampler = decode_sampler
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.epochs = epochs
        self.schedule = schedule
        self.swa_lr = swa_lr

        # Save additional hparams for logging purposes, if available
        self.max_seq_len = max_seq_len if max_seq_len is not None else "Unknown"
        self.batch_size = batch_size if batch_size is not None else "Unknown"
        self.augment = augment if augment is not None else "Unknown"
        self.warm_up_steps = warm_up_steps if warm_up_steps is not None else "None"
        self.acc_batches = acc_batches if acc_batches is not None else "Unknown"

        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        self.save_hyperparameters(
            "lr",
            "weight_decay",
            "num_steps",
            "epochs",
            "schedule",
            "swa_lr",
            "max_seq_len", 
            "batch_size",
            "clip_grad",
            "augment",
            "warm_up_steps",
            "acc_batches"
        )

        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 5

        self.model.sampler = self.sampler
        self.model.num_beams = self.num_beams
        #self.model.device = self.device
        self.model.to(self.device)

        self.use_swa = False
        self.swa_model = None

        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)

    def forward(self, x, swa=False):
        """ Apply molecules to model to produce a set of output probability distributions

        Args:
            x (dict {
                "encoder_input": tensor of shape (src_len, batch_size)
                "decoder_input": tensor of shape (tgt_len, batch_size)
                "encoder_pad_mask": bool tensor of shape (src_len, batch_size)
                "decoder_pad_mask": bool tensor of shape (tgt_len, batch_size)
            })

        Returns:
            Output from model (dict {
                "token_output": tensor of shape (tgt_len, batch_size, vocab_size)
                "model_output": tensor of shape (tgt_len, batch_size, d_model)
            })
        """

        enc_tokens = x["encoder_input"]
        enc_mask = x["encoder_pad_mask"]
        dec_tokens = x["decoder_input"]
        dec_mask = x["decoder_pad_mask"]

        # Replicate dict so that the model does not see the target tensor
        model_input = {
            "encoder_input": enc_tokens,
            "encoder_pad_mask": enc_mask,
            "decoder_input": dec_tokens,
            "decoder_pad_mask": dec_mask
        }

        if swa:
            output = self.swa_model(model_input)
        else:
            output = self.model(model_input)

        return output

    def training_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        swa = self.swa_lr is not None and self.use_swa
        model_output = self.forward(batch, swa=swa)

        loss = self._calc_loss(batch, model_output)
        char_acc = self._calc_char_acc(batch, model_output)

        target_smiles = batch["target_smiles"]
        mols, log_lhs = self.model.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol, target_smiles)

        self.log("val_loss", loss)
        self.log("val_accuracy", char_acc)
        self.log("val_molecular_accuracy", metrics["accuracy"], prog_bar=True)
        self.log("val_invalid", metrics["invalid"])

    def test_step(self, batch, batch_idx):
        swa = self.swa_lr is not None and self.use_swa
        model_output = self.forward(batch, swa=swa)

        loss = self._calc_loss(batch, model_output)
        char_acc = self._calc_char_acc(batch, model_output)

        target_smiles = batch["target_smiles"]
        mols, log_lhs = self.model.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mols, target_smiles)

        test_outputs = {
            "test_loss": loss,
            "test_accuracy": char_acc,
            "test_invalid_smiles": metrics["invalid"]
        }

        if self.test_sampling_alg == "greedy":
            test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        elif self.test_sampling_alg == "beam":
            test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
            test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
            test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
            # test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]
            # test_outputs["test_molecular_top_20_accuracy"] = metrics["top_20_accuracy"]

        else:
            raise ValueError(f"Unknown test sampling algorithm, {self.test_sampling_alg}")

        return test_outputs

    def test_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def configure_optimizers(self):
        params = self.model.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.998))

        if self.swa_lr is not None and self.use_swa:
            print("Using SWA LR schedule")
            sch = SWALR(optim, anneal_epochs=5, swa_lr=self.swa_lr)

        else:
            if self.schedule == "const":
                print("Using constant LR schedule.")
                sch = LambdaLR(optim, lr_lambda=lambda epoch: 1)

            elif self.schedule == "cycle":
                print("Using cyclical LR schedule.")
                cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
                sch = {"scheduler": cycle_sch, "interval": "step"}

            elif self.schedule == "transformer":
                print("Using original transformer schedule.")
                trans_sch = _FuncLR(optim, lr_lambda=self._transformer_lr)
                sch = {"scheduler": trans_sch, "interval": "step"}

            else:
                raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.model.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def on_train_epoch_end(self, outputs):
        if self.swa_lr is not None and self.use_swa:
            self.swa_model.update_parameters(self.model)

    def set_swa(self):
        self.use_swa = True
        self.swa_model = AveragedModel(self.model)

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val)

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the cross-entropy loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor)
        """

        target = batch_input["target"]
        target_mask = batch_input["target_pad_mask"]
        token_output = model_output["token_output"]

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        num_tokens = (~target_mask).sum()
        loss = loss.sum() / num_tokens

        return loss

    def _calc_char_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        token_output = model_output["token_output"]

        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)

        num_correct = correct_ids.sum().cpu().detach().item()
        total = correct_ids.numel()

        accuracy = num_correct / total
        return accuracy


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Helper Classes ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        # Self attention block 
        query = self.norm1(tgt)
        query = self.self_attn(query, query, query, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(att, memory, memory, attn_mask=memory_mask, 
                key_padding_mask=memory_key_padding_mask)[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out
