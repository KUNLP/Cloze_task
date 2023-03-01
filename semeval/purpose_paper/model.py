import argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import torch


class SemEvalTransformer(pl.LightningModule):
    def __init__(
            self,
            args: argparse.Namespace,
            learning_rate: float,
            weight_decay: float,
            adam_epsilon: float,
            warmup_steps: int,
            train_batch_size: int,
            eval_batch_size: int,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model_name_or_path = args.model_name_or_path
        self.num_labels = args.num_labels
        self.config = AutoConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, config=self.config)
        self.metric = accuracy_score

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        train_loss = outputs.loss
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)

        self.log('train_loss', train_loss, on_step=True, logger=True)
        return {"loss": train_loss, "prediction": prediction, "labels": labels}

    def training_epoch_end(self, outputs):
        prediction = torch.cat([x["prediction"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        acc = self.metric(y_true=labels, y_pred=prediction)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)

        #self.log('val_loss', val_loss, on_step=True, logger=True)
        return {"loss": val_loss, "prediction": prediction, "labels": labels}

    def validation_epoch_end(self, outputs):
        prediction = torch.cat([x["prediction"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, on_epoch=True, logger=True)
        acc = self.metric(y_true=labels, y_pred=prediction)
        self.log("val_acc", acc, on_epoch=True, logger=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [optimizer], [scheduler]

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parent_parser.add_argument("--learning_rate", type=float, default=2e-5)
    #     parent_parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    #     parent_parser.add_argument("--warmup_steps", type=int, default=0)
    #     parent_parser.add_argument("--weight_decay", type=float, default=0.0)
    #     return parent_parser
