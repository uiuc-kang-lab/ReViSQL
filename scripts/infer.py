import chz
from tinker_cookbook.recipes.sql_rl.sql_env import BIRDDataset
from tinker_cookbook.rl import train
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
import os
import tinker
import asyncio
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from dotenv import load_dotenv
from tinker_cookbook.utils.ml_log import dump_config
import wandb

load_dotenv()

@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    max_output_tokens_per_turn: int = 3000
    max_turns: int = 5
    data_path: str | None = None
    db_path: str | None = None
    log_path: str | None = None
    run_name: str
    output_path: str
    timeout: int = 30
    sampler_path: str
    test_data: str = "val_test.parquet"
    use_tool_call_template: str = "default"
    temperature: float = 1
    repeat: int = 1
    sql_engine: str = "SQLite"
    # Use the evidence-supervision system prompt at inference to match training.
    use_evidence_supervision_prompt: bool = True


async def build_config(cli_config: CLIConfig) -> tuple[RLTestSetEvaluator, tinker.SamplingClient]:
    if cli_config.output_path is not None:
        os.makedirs(cli_config.output_path, exist_ok=True)
    model_name = cli_config.model_name
    renderer_name = get_recommended_renderer_name(model_name)
    os.makedirs(cli_config.output_path, exist_ok=True)
    os.makedirs(cli_config.log_path, exist_ok=True)

    test_dataset = BIRDDataset(
        batch_size=64,
        group_size=1,
        renderer=get_renderer(renderer_name, get_tokenizer(model_name)),
        data_path=f"{cli_config.data_path}/{cli_config.test_data}",
        db_modification_script_path=None,
        timeout=cli_config.timeout,
        model_name=cli_config.model_name if cli_config.use_tool_call_template == "default" else "N/A",
        db_path=cli_config.db_path,
        split="test",
        n_epochs=cli_config.repeat,
        dump_path=cli_config.output_path,
        max_turns=cli_config.max_turns,
        max_output_tokens_per_turn=cli_config.max_output_tokens_per_turn,
        sql_engine=cli_config.sql_engine,
        use_evidence_supervision_prompt=cli_config.use_evidence_supervision_prompt,
    )

    print("Creating service client...")
    service_client = tinker.ServiceClient(base_url=None)
    print("Creating training client...")
    training_client = await service_client.create_lora_training_client_async(
        cli_config.model_name, rank=32
    )
    print("Creating sampling client...")
    sampling_client = training_client.create_sampling_client(cli_config.sampler_path)

    return RLTestSetEvaluator(test_dataset, max_tokens=cli_config.max_output_tokens_per_turn, dump_path=cli_config.output_path, temperature=cli_config.temperature, name=cli_config.run_name, save_traces=True), sampling_client

async def main():
    cli_config = chz.entrypoint(CLIConfig)
    run = wandb.init(
        project="revisql",
        config=dump_config(cli_config),
        dir=f"{cli_config.log_path}/{cli_config.run_name}",
        name=cli_config.run_name,
    )
    evaluator, sampling_client = await build_config(cli_config)
    wandb.termlog("Starting evaluation...")
    _ = await evaluator(sampling_client)
    run.finish()


if __name__ == "__main__":
    asyncio.run(main())