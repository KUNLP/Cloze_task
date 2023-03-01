from pytorch_lightning import Trainer, seed_everything
import os
from purpose_paper.data import SemEvalDataModule
from purpose_paper.model import SemEvalTransformer
import argparse
from attrdict import AttrDict

import torch
from pytorch_lightning.loggers import WandbLogger

AVAIL_GPUS = min(1, torch.cuda.device_count())
wandb_logger = WandbLogger(name='electra base', project='pytorchlightning')
seed_everything(0)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--model_name_or_path", type=str, default='google/electra-base-discriminator')

    args.add_argument("--learning_rate", type=float, default=3e-5)
    args.add_argument("--adam_epsilon", type=float, default=1e-8)
    args.add_argument("--warmup_steps", type=int, default=0)
    args.add_argument("--weight_decay", type=float, default=0.0)

    args.add_argument("--train_batch_size", type=int, default=32)
    args.add_argument("--eval_batch_size", type=int, default=32)
    args.add_argument("--test_batch_size", type=int, default=32)

    args.add_argument("--num_labels", type=int, default=3)
    args.add_argument("--max_seq_length", type=int, default=256)

    # Trainer
    args.add_argument("--max_epochs", type=int, default=20)
    args.add_argument("--gpus", type=int, default=AVAIL_GPUS)
    args.add_argument("--logger", default=wandb_logger)
    args.add_argument("--profiler", type=str, default="simple")

    args = args.parse_args()

    dm = SemEvalDataModule(args)
    model = SemEvalTransformer(
        args=args,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    trainer = Trainer(
        max_epochs=20,
        gpus=AVAIL_GPUS,
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=dm)
