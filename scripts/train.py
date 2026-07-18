import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.sql_rl.sql_env import SQLEnv, BIRDDatasetBuilder
from tinker_cookbook.rl import train
from tinker_cookbook.model_info import get_recommended_renderer_name
import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    group_size: int = 8
    batch_size: int = 64
    learning_rate: float = 5e-5
    max_output_tokens_per_turn: int = 3000
    max_input_tokens: int = 32768
    max_turns: int = 5
    eval_every: int = 10
    save_every: int = 20
    wandb_project: str | None = "revisql"
    wandb_name: str | None = None
    log_path: str | None = None
    train_data_path: str | None = None
    test_data_path: str | None = None
    db_path: str | None = None
    db_modification_script_path: str | None = None
    checkpoint_path: str | None = None
    add_noise: str | None = None
    timeout: int = 30
    n_epochs: int = 1
    num_data: int = -1
    use_convo_prefix: bool = True
    curriculum_learning: bool = False
    dynamic_sampling: bool = False
    renderer_name: str = "default"
    dump_test_results: bool = False
    dump_dir: str | None = None
    do_async: bool = False
    async_max_steps_off_policy: int = 100
    loss_fn: Literal["ppo", "cispo", "importance_sampling", "drpo", "sapo", "sapo_seq"] = "ppo"
    loss_clip_high: float = 1.2
    loss_clip_low: float = 0.8
    dro_beta: float = 0.05
    tau_pos: float = 0.1
    tau_neg: float = 0.1
    lora_rank: int = 32
    sql_engine: Literal["SQLite", "Snowflake"] = "SQLite"
    # Reward configuration (default ON). Requires `uv run --extra verieql`.
    use_verieql: bool = True
    use_evidence_supervision_prompt: bool = True
    use_evidence_supervision_penalty: bool = True

def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-{cli_config.group_size}group-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"./outputs/tinker/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    if cli_config.dump_test_results and cli_config.dump_dir is not None:
        os.makedirs(cli_config.dump_dir, exist_ok=True)

    dataset_builder = BIRDDatasetBuilder(
        batch_size=cli_config.batch_size,
        renderer_name=get_recommended_renderer_name(model_name) if cli_config.renderer_name == "default" else cli_config.renderer_name,
        train_group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        train_data_path=cli_config.train_data_path,
        test_data_path=cli_config.test_data_path,
        db_path=cli_config.db_path,
        timeout=cli_config.timeout,
        add_noise=cli_config.add_noise,
        n_epochs=cli_config.n_epochs,
        db_modification_script_path=cli_config.db_modification_script_path,
        num_data=cli_config.num_data,
        use_convo_prefix=cli_config.use_convo_prefix,
        max_output_tokens_per_turn=cli_config.max_output_tokens_per_turn,
        curriculum_learning=cli_config.curriculum_learning,
        max_turns=cli_config.max_turns,
        max_input_tokens=cli_config.max_input_tokens,
        sql_engine=cli_config.sql_engine,
        use_verieql=cli_config.use_verieql,
        use_evidence_supervision_prompt=cli_config.use_evidence_supervision_prompt,
        use_evidence_supervision_penalty=cli_config.use_evidence_supervision_penalty,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_output_tokens_per_turn,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        save_every=cli_config.save_every,
        load_checkpoint_path=cli_config.checkpoint_path,
        remove_constant_reward_groups=cli_config.dynamic_sampling,
        use_dynamic_sampling=cli_config.dynamic_sampling,
        dump_test_results=cli_config.dump_test_results,
        dump_dir=cli_config.dump_dir,
        async_config=train.AsyncConfig(
            max_steps_off_policy=cli_config.async_max_steps_off_policy,
            groups_per_batch=cli_config.batch_size,
        ) if cli_config.do_async else None,
        loss_fn=cli_config.loss_fn,
        loss_clip_high=cli_config.loss_clip_high,
        loss_clip_low=cli_config.loss_clip_low,
        dro_beta=cli_config.dro_beta,
        tau_neg=cli_config.tau_neg,
        tau_pos=cli_config.tau_pos,
        lora_rank=cli_config.lora_rank,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
