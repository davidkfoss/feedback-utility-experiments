import dataclasses, hashlib, random
from types import SimpleNamespace
import torch
from experiments import task_agnews
from experiments.utils.flops import FlopAccumulator


def run_toy(common):
    config = dataclasses.replace(
        task_agnews.parse_args(), episode_length=2, lr=0.01, epochs=1
    )
    common.set_seed(17)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 4)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, input_ids, labels=None):
            logits = self.layer(self.dropout(input_ids))
            return SimpleNamespace(
                logits=logits,
                loss=None
                if labels is None
                else torch.nn.functional.cross_entropy(logits, labels),
            )

    model = Toy()
    components = common.build_method_components(config, model)
    batches = [
        {
            "input_ids": torch.tensor([[0.1, 0.3, -0.4], [0.7, -0.8, 0.2]]),
            "labels": torch.tensor([i % 4, (i + 1) % 4]),
        }
        for i in range(5)
    ]
    scheduler = common.build_lr_scheduler(
        config, components.optimizer, total_training_steps=5
    )
    flop = FlopAccumulator()
    metrics = common.train_one_epoch(
        model=model,
        loader=batches,
        device=torch.device("cpu"),
        components=components,
        global_step=0,
        flop_accumulator=flop,
        lr_scheduler=scheduler,
    )
    if components.episode_manager:
        components.episode_manager.finalize()
    return {
        "metrics": metrics,
        "weights": {k: v.tolist() for k, v in model.state_dict().items()},
        "logs": components.episode_manager.get_logs()
        if components.episode_manager
        else None,
        "torch_rng": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        "python_rng": repr(random.getstate()),
        "lr": components.optimizer.param_groups[0]["lr"],
    }
