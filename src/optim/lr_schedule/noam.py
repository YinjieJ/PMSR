from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class NoamRootLRScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is cfg.lr"
        },
    )
    lr: List[float] = II("optimization.lr")
    encoder_embed_dim: float = II("model.encoder_embed_dim")
    



@register_lr_scheduler("noam", dataclass=NoamRootLRScheduleConfig)
class NoamSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, cfg: NoamRootLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with noam."
                " Consider --lr-scheduler=fixed instead."
            )
        # warmup_end_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        

        # # linearly warmup for the first cfg.warmup_updates
        # self.lr_step = (warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_updates

        # # then, decay prop. to the inverse square root of the update number
        self.decay_factor = cfg.lr[0] * cfg.encoder_embed_dim ** (-0.5)
        if cfg.warmup_init_lr < 0:
            cfg.warmup_init_lr = 0 if cfg.warmup_updates > 0 else self.decay_factor

        # initial learning rate
        self.lr = cfg.warmup_init_lr
        self.warmup_update = cfg.warmup_updates if cfg.warmup_updates > 0 else 1
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates:
            self.lr = self.decay_factor * min(num_updates ** (-0.5), num_updates * self.warmup_update ** (-1.5))
            self.optimizer.set_lr(self.lr)
        else:
            self.optimizer.set_lr(self.lr)
        return self.lr
