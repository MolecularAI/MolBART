import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models.util import FuncLR
from molbart.models.pre_train import BARTModel


class ReactionBART(BARTModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule,
        dropout=0.1,
        warm_up_steps=None,
        **kwargs
    ):
        super().__init__(
            decode_sampler,
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            dropout,
            schedule=schedule,
            warm_up_steps=warm_up_steps,
            **kwargs
        )

        self.schedule = schedule
        self.warm_up_steps = warm_up_steps

        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 5

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)

        loss = self._calc_loss(batch, model_output)
        char_acc = self._calc_char_acc(batch, model_output)

        target_smiles = batch["target_smiles"]
        mols, _ = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mols, target_smiles)

        # Log for prog bar only
        self.log("mol_acc", metrics["accuracy"], prog_bar=True, logger=False)

        val_outputs = {
            "val_loss": loss.item(),
            "val_token_acc": char_acc,
            "val_molecular_accuracy": metrics["accuracy"],
            "val_invalid": metrics["invalid"]
        }
        return val_outputs
    def validation_epoch_end(self, outputs):
        #sys.stderr.write(str(outputs))
        val_loss =outputs[0]['val_loss']# torch.stack([ou['val_loss'] for x in outputs]).mean()
        val_token_acc = outputs[0]['val_token_acc']#torch.stack([x['val_token_acc'] for x in outputs]).mean()
        #val_perplexity = outputs[0]['val_perplexity']#torch.stack([x['val_perplexity'] for x in outputs]).mean()
        val_molecular_accuracy = outputs[0]['val_molecular_accuracy']# torch.stack(['val_molecular_accuracy'] for x in outputs).mean()
        val_invalid_smiles = outputs[0]['val_invalid']#torch.stack(['val_invalid_smiles'] for x in outputs).mean()
        log = {'val_loss': val_loss, 'val_token_acc': val_token_acc, 'val_molecular_accuracy': val_molecular_accuracy, 'val_invalid': val_invalid_smiles}
        self._log_dict(log)
        # avg_outputs = self._avg_dicts(outputs)
        # self._log_dict(avg_outputs)
    def test_step(self, batch, batch_idx):
        model_output = self.forward(batch)

        loss = self._calc_loss(batch, model_output)
        char_acc = self._calc_char_acc(batch, model_output)

        target_smiles = batch["target_smiles"]
        mols, _ = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mols, target_smiles)

        test_outputs = {
            "test_loss": loss,
            "test_token_acc": char_acc,
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
        params = self.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.998))

        if self.schedule == "const":
            print("Using constant LR schedule.")
            sch = LambdaLR(optim, lr_lambda=lambda epoch: 1)

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

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

        loss = self._calc_mask_loss(token_output, target, target_mask)
        return loss
