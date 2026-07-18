import json
import pandas as pd
import asyncio
import chz
from tinker import ModelInput
from tinker_cookbook.completers import (
    StopCondition,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, ToolCall, ContentPart
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.sql_rl.sql_utils import verify_format_and_extract, execute_sql_wrapper_single
from tinker_cookbook.recipes.sql_rl.grader import grade
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.recipes.sql_rl.prompts import system_prompt, SQL_TOOLS
from tinker_cookbook.recipes.sql_rl.prompts_evidence_supervision import bird_system_prompt as system_prompt_evidence_supervision_bird
from tinker_cookbook.recipes.sql_rl.verieql_grader import grade_with_verieql
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Literal, cast, Tuple, Any
from functools import partial
import math
import re
import os
import logging
from typing import List
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import wandb

logger = logging.getLogger(__name__)

qwen_one_shot_prompt = [
    Message(role="user", content="Database Engine:\n{engine}\n\nDatabase Schema:\nCREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\nExternal Knowledge:\npig is the species of.\n\nQuestion:\nhow many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.\n<function_call>{\"name\": \"sql\", \"args\": {\"query\": \"SELECT COUNT(*) FROM animals WHERE species = 'pig';\"}}</function_call>"),
    Message(role="user", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

general_one_shot_prompt = [
    Message(role="user", content="Database Engine:\n{engine}\n\nDatabase Schema:\nCREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\nExternal Knowledge:\npig is the species of.\n\nQuestion:\nhow many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.\n<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>"),
    Message(role="user", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

kimi_one_shot_prompt = [
    Message(role="user", content="Database Engine:\n{engine}\n\nDatabase Schema:\nCREATE TABLE animals (\n`id` integer, -- ID of the animal.\n`species` text, -- species of the animal.\n`age` integer, -- age of the animal.\n`name` text, -- name of the animal.\nprimary key (id)\n);\n\nExternal Knowledge:\npig is the species of.\n\nQuestion:\nhow many pigs are in the farm?"),
    Message(role="assistant", content="I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.", tool_calls=[ToolCall(function=ToolCall.FunctionBody(name="execute_sql_query", arguments='{"query": "SELECT COUNT(*) FROM animals WHERE species = \'pig\';"}'), id="functions.execute_sql_query:0")]),
    Message(role="tool", content="+----------+\n| COUNT(*) |\n+----------+\n|   12     |\n+----------+\n", tool_call_id="functions.execute_sql_query:0", name="execute_sql_query"),
    Message(role="assistant", content="The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.\n<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>")
]

def shorten_overlong_cells(df: pd.DataFrame, limit: int = 200):
    def shorten_cell(cell):
        cell_str = str(cell)
        if len(cell_str) > limit:
            return cell_str[:limit] + "...(truncated)"
        else:
            return cell_str

    return df.map(shorten_cell)
    

def convert_toolcall_to_dict(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "name": tool_call.function.name,
        "arguments": tool_call.function.arguments,
        "id": tool_call.id
    }
    
def convert_content_to_dict(content: str | list[ContentPart]) -> dict[str, Any]:
    if isinstance(content, str):
        return {"type": "text", "text": content}
    elif isinstance(content, list):
        contents = []
        for part in content:
            if part["type"] == "text":
                contents.append({"type": "text", "text": part["text"]})
            elif part["type"] == "thinking":
                contents.append({"type": "thinking", "thinking": part["thinking"]})
            elif part["type"] == "tool_call":
                contents.append({
                    "type": "tool_call",
                    "name": part["tool_call"].function.name,
                    "arguments": part["tool_call"].function.arguments,
                    "id": part["tool_call"].id
                })
            elif part["type"] == "unparsed_tool_call":
                contents.append({
                    "type": "unparsed_tool_call",
                    "raw_text": part["raw_text"],
                    "error": part["error"]
                })
            elif part["type"] == "image":
                contents.append({
                    "type": "image",
                })
            else:
                contents.append({"type": part["type"]})
        return {"type": "composite", "parts": contents}
    else:
        raise ValueError("Content must be a string or a list of ContentPart")

class SQLEnv(Env):
    def __init__(self, question_id: str, question: str, external_knowledge: str | None, gold_answer: int, grading_method: str, renderer: Renderer, db_file, timeout, db_modification_script: str | None, dump_path: str | None = None, use_convo_prefix: bool = True, max_output_tokens_per_turn: int = 3072,
    max_input_tokens: int = 32768, model_name: str = "qwen", max_turns: int = 5,
    sql_engine: str = "SQLite", use_verieql: bool = False,
    use_evidence_supervision_prompt: bool = False, use_evidence_supervision_penalty: bool = False):

        assert sql_engine in ("SQLite", "Snowflake"), "Only SQLite and Snowflake are supported currently"

        self.renderer: Renderer = renderer
        self.turns: list[Message] = []
        self.external_knowledge = external_knowledge
        self.gold_answer: int = gold_answer
        self.db_file = db_file
        self.timeout = timeout
        self.num_turn = 0
        self.question = question
        self.question_id = question_id
        self.db_modification_script = db_modification_script
        self.dump_path = dump_path
        self.grading_method = grading_method
        self.use_convo_prefix = use_convo_prefix
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_input_tokens = max_input_tokens
        self.model_name = model_name
        self.sql_engine = sql_engine
        self.use_verieql = use_verieql
        self.use_evidence_supervision_prompt = use_evidence_supervision_prompt
        self.use_evidence_supervision_penalty = use_evidence_supervision_penalty
        # Track whether each evidence-supervision penalty fired during the episode.
        # Surfaced in StepResult.metrics on the terminal step so batch means give per-episode rates.
        self._requirement_penalty_applied: bool = False
        self._verification_penalty_applied: bool = False

        # Count non-empty external-knowledge pieces (separated by ';') for the
        # evidence-supervision requirement/verification checks.
        external_knowledge_list = external_knowledge.split(";") if external_knowledge is not None else []
        self.n_effective_knowledge = 0
        for knowledge in external_knowledge_list:
            if knowledge.strip().strip(";") == '':
                continue
            self.n_effective_knowledge += 1

        if "Qwen3" in self.model_name:
            one_shot_prompt = qwen_one_shot_prompt
        elif "Kimi" in self.model_name or "gpt" in self.model_name:
            one_shot_prompt = kimi_one_shot_prompt
        else:
            one_shot_prompt = general_one_shot_prompt

        one_shot_prompt[0]['content'] = one_shot_prompt[0]['content'].format(engine=self.sql_engine)
        if self.use_evidence_supervision_prompt:
            self.system_prompt = system_prompt_evidence_supervision_bird.format(engine=self.sql_engine)
        else:
            self.system_prompt = system_prompt.format(engine=self.sql_engine)
        self.convo_prefix = one_shot_prompt
        
        if self.sql_engine == "Snowflake":
            # adjust max turns based on token limit for snowflake
            self.max_turns = max(min(max_turns, (self.max_input_tokens - self._obs.length) // self.max_output_tokens_per_turn - 1), 2)  # at least 2 turns for snowflake to allow for some back and forth
            # print(f"Start length: {self._obs.length}, max turns adjusted to: {self.max_turns} for Snowflake")
        else:
            self.max_turns = max_turns
        
        self.traces = [
            {
                "role": "system",
                "tools": SQL_TOOLS,
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.question
            }
        ]

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @property
    def _obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""

        if self.use_convo_prefix:
            convo = self.renderer.create_conversation_prefix_with_tools(tools=SQL_TOOLS, system_prompt=self.system_prompt) + self.convo_prefix + [Message(role="user", content=self.question)] + self.turns
        elif not self.use_convo_prefix:
            convo = self.renderer.create_conversation_prefix_with_tools(tools=SQL_TOOLS, system_prompt=self.system_prompt) + [Message(role="user", content=self.question)] + self.turns

        return self.renderer.build_generation_prompt(convo)

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._obs, self.stop_condition

    def _parse_tool_calls(self, tool_calls: List[ToolCall] = []) -> Tuple[List[str|None], List[str|None]]:
        sqls = []
        errors = []
        for tool_call in tool_calls:
            arguments = tool_call.function.arguments
            try:
                sql = json.loads(arguments)
                next_sql = sql["query"]
                next_error = None
            except Exception as e:
                next_sql = None
                next_error = f"Error in parsing tool call arguments: {e}"
            sqls.append(next_sql)
            errors.append(next_error)
        return sqls, errors
    
    def _get_last_text_part(self, content: str | list[ContentPart]) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            for part in reversed(content):
                if part["type"] == "text":
                    return part["text"]
        return ""

    def _get_all_text_part(self, content: str | list[ContentPart]) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if part["type"] == "text":
                    texts.append(part["text"])
            return "\n".join(texts)
        return ""

    _REQUIREMENT_RE = re.compile(r"Requirement of external knowledge\s+\d+\s*\(", re.IGNORECASE)
    _VERIFICATION_RE = re.compile(r"Verification of external knowledge\s+\d+\s*\(", re.IGNORECASE)

    def _has_evidence_supervision(self, n_knowledge: int, message_text: str, mode: str) -> bool:
        if mode == "check":
            return len(self._REQUIREMENT_RE.findall(message_text)) >= n_knowledge
        elif mode == "verify":
            return len(self._VERIFICATION_RE.findall(message_text)) >= n_knowledge
        else:
            raise ValueError("Mode must be either 'check' or 'verify'")

    def _check_missing_requirement(self) -> bool:
        """True if no assistant turn so far contains the enumerated requirement block."""
        if self.n_effective_knowledge == 0:
            return False
        for turn in self.turns:
            if turn["role"] == "assistant":
                message_text = self._get_all_text_part(turn["content"])
                if self._has_evidence_supervision(self.n_effective_knowledge, message_text, mode="check"):
                    return False
        return True

    def _check_missing_verification(self) -> bool:
        """True if the final assistant message lacks the enumerated verification block."""
        if self.n_effective_knowledge == 0:
            return False
        # find the last assistant message
        last_message = None
        for turn in reversed(self.turns):
            if turn["role"] == "assistant":
                last_message = turn
                break
        if last_message is None:
            return False
        message_text = self._get_all_text_part(last_message["content"])
        return not self._has_evidence_supervision(self.n_effective_knowledge, message_text, mode="verify")

    async def _get_user_turn(self, action_text: str, tool_calls: List[ToolCall] = []) -> tuple[List[Message]|None, float, float]:
        # Returns (user_turn_messages_or_None, outcome_reward, prm_reward).
        # prm_reward is the reward attributed to a non-terminal step; it is always 0.0
        # here (no process reward model) but is kept so the evidence-supervision penalty
        # can dock an intermediate step in step().
        # check if there is a solution
        if not action_text.endswith('</solution>'):
            # this means this turn is an intermediate step involving tool calls
            messages = []
            sqls, errors = self._parse_tool_calls(tool_calls)
            index = 0
            for sql, error in zip(sqls, errors):
                if sql is None and error is None:
                    assert False, "Should not reach here"
                elif sql is None and error is not None:
                    messages.append(Message(role="tool", content=f"Your previous action is invalid due to error: {error}. Follow the format of outputting a sql tool call.", tool_call_id=tool_calls[index].id, name="execute_sql_query"))
                    index += 1
                    continue

                assert sql is not None and error is None

                sql_output = await execute_sql_wrapper_single(self.db_file, sql, self.timeout, engine=self.sql_engine)
                pred_results, error = sql_output

                if pred_results is None:
                    messages.append(Message(role="tool", content=error, tool_call_id=tool_calls[index].id, name="execute_sql_query"))
                else:
                    df = pd.DataFrame(pred_results)
                    res = df.to_string(index=False)
                    if len(df) > 50:
                        # just truncate
                        truncated_df = df.head(50)
                        if self.sql_engine == "Snowflake" or self.question_id.startswith("local"):
                            # truncate rows
                            truncated_df = shorten_overlong_cells(truncated_df, limit=200)
                        res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                            index=False
                        )  # or index=True if you want row numbers
                    else:
                        if self.sql_engine == "Snowflake" or self.question_id.startswith("local"):
                            res = shorten_overlong_cells(df, limit=200).to_string(index=False)
                        res = "SQL execution results: " + res
                    # print(f"SQL output: {res}")
                    response = f"{res}\nYou have {self.max_turns - self.num_turn} turns left to complete the task."
                    
                    messages.append(Message(role="tool", content=response, tool_call_id=tool_calls[index].id, name="execute_sql_query"))

                index += 1

            if len(messages) == 0:
                messages.append(Message(role="user", content="Your previous action is invalid. Follow the format of outputting a sql tool call or a final solution."))
            return messages, 0.0, 0.0
        else:
            is_valid, _, pred_sql, _ = verify_format_and_extract(action_text)

            if not is_valid:
                return [Message(role="user", content="Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again.")], -1.0, 0.0

            # if self.dump_path is not None:
            #     with open(os.path.join(self.dump_path, f"{self.question_id}_pred.sql"), "w") as f:
            #         f.write(pred_sql.replace("\n", " "))
                

            if self.sql_engine == "Snowflake":
                wandb.termlog(f"Prediction for Snowflake Problem {self.question_id}: {pred_sql}")
                return None, 0.0, 0.0  # skip grading for snowflake for now
            elif "local" in self.question_id:
                wandb.termlog(f"Prediction for Spider2SQLite Problem {self.question_id}: {pred_sql}")
                return None, 0.0, 0.0

            pred = await execute_sql_wrapper_single(self.db_file, pred_sql, self.timeout)
            ref = await execute_sql_wrapper_single(self.db_file, self.gold_answer, self.timeout)

            pred_results, error = pred
            gt_results, gt_error = ref

            if pred_results is None:
                wandb.termlog(f"Fail to execute the prediction for Problem {self.question_id}: {pred_sql}")
                return None, 0.0, 0.0
            elif gt_results is None:
                wandb.termlog(f"Fail to execute the groundtruth for Problem {self.question_id}: {self.gold_answer}")
                return None, 0.0, 0.0
            else:
                if grade(gt_results, pred_results, self.grading_method)[0]:
                    # Execution grading says correct; check formal equivalence with VeriEQL.
                    # VeriEQL rejection gives a soft reward (0.8) to hedge against false negatives.
                    if self.use_verieql and self.sql_engine == "SQLite":
                        verieql_result = await grade_with_verieql(
                            self.db_file, pred_sql, self.gold_answer, timeout=30.0
                        )
                        if verieql_result is False:
                            wandb.termlog(
                                f"[VeriEQL] non-equivalent prediction for Problem {self.question_id}: {pred_sql}"
                            )
                            return None, 0.8, 0.0
                    wandb.termlog(f"Correct prediction for Problem {self.question_id}: {pred_sql}")
                    return None, 1.0, 0.0
                else:
                    wandb.termlog(f"Wrong prediction for Problem {self.question_id}: {pred_sql}")
                    return None, 0.0, 0.0


    def _is_done(self, action: str) -> bool:
        if self.num_turn == self.max_turns:
            return True
        return "<solution>" in action and "</solution>" in action


    async def step(self, action: Action) -> StepResult:
        self.num_turn += 1

        # step 1: parse the action tokens into a message
        # this step is specific to our library, but usually templated, so you can just copy it.
        (action_message, _parse_success) = self.renderer.parse_response(action)
        
        self.traces.append({
            "role": action_message["role"],
            "content": convert_content_to_dict(action_message["content"]),
            "tool_calls": [convert_toolcall_to_dict(tc) for tc in action_message["tool_calls"]] if "tool_calls" in action_message else []
        })

        # step 2: based on the string answer, we compute the reward and the user turn.
        user_turn, reward, prm_reward = await self._get_user_turn(
            self._get_last_text_part(action_message["content"]),
            action_message["tool_calls"] if "tool_calls" in action_message else []
        )

        # step 3: update the conversation history
        self.turns.append({"role": "assistant", "content": action_message["content"]})
        if "tool_calls" in action_message and action_message["tool_calls"] != []:
            self.turns[-1]["tool_calls"] = action_message["tool_calls"]
    
        # print("=" * 100)
        # print(f"Action: {self._get_last_text_part(action_message['content'])}")
        # if "tool_calls" in action_message:
        #     for tool_call in action_message["tool_calls"]:
        #         print(f"Tool Call (id={tool_call.id}): {tool_call.function.name} with arguments {tool_call.function.arguments}")

        if user_turn is not None:
            self.turns += user_turn
            # print(f"Next user turn:")
            for user_turn_1 in user_turn:
                self.traces.append({
                    "role": user_turn_1["role"],
                    "content": user_turn_1["content"],
                    "tool_call_id": user_turn_1.get("tool_call_id")
                })
                # if user_turn_1['role'] == 'tool':
                #     print(f"Tool Call ID: {user_turn_1.get('tool_call_id')}")
                # print(f"Role: {user_turn_1['role']}, Content: {user_turn_1['content']}")
        else:
            # no user turn means episode is done, we can print the reward for debugging
            self.traces.append({
                "role": "system",
                "content": f"Episode done with reward: {reward}"
            })
        # Determine episode termination: natural end via _is_done, or forced end via context overflow.
        last_action_text = self._get_last_text_part(action_message['content'])
        emitted_solution = "<solution>" in last_action_text and "</solution>" in last_action_text
        episode_done = self._is_done(last_action_text)

        if self._obs.length + self.max_output_tokens_per_turn > self.max_input_tokens:
            episode_done = True
            print("Observation too long, marking episode as done.")
            print(f"Observation length: {self._obs.length}, max input tokens: {self.max_input_tokens}, max output tokens per turn: {self.max_output_tokens_per_turn}")
            print(self.traces)

        # Evidence-supervision penalty runs with the authoritative episode_done (post-overflow)
        # and uses emitted_solution to gate checks that only make sense at solution time.
        if self.use_evidence_supervision_penalty:
            if emitted_solution and self.num_turn == 1:
                # Easy-question soft rule: a one-turn solution only needs one of the two blocks.
                if self._check_missing_requirement() and self._check_missing_verification():
                    reward -= 0.1
                    self._requirement_penalty_applied = True
                    self._verification_penalty_applied = True
            elif self.num_turn == 2 and self._check_missing_requirement():
                # Requirement translation must appear by turn 2; fires once per episode.
                if episode_done:
                    reward -= 0.1
                else:
                    prm_reward -= 0.1
                self._requirement_penalty_applied = True
            # Verification only applies when the model actually commits a solution.
            if emitted_solution and self.num_turn > 1 and self._check_missing_verification():
                reward -= 0.1
                self._verification_penalty_applied = True

        # step 4: return the step result
        step_metrics: dict[str, float | int] = {}
        if episode_done and self.use_evidence_supervision_penalty and self.n_effective_knowledge > 0:
            step_metrics["evidence_supervision/requirement_penalty"] = float(self._requirement_penalty_applied)
            step_metrics["evidence_supervision/verification_penalty"] = float(self._verification_penalty_applied)
            step_metrics["evidence_supervision/any_penalty"] = float(
                self._requirement_penalty_applied or self._verification_penalty_applied
            )

        step_result = StepResult(
            next_observation=self._obs,
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward=reward if episode_done else prm_reward,
            metrics=step_metrics,
        )

        return step_result

class BIRDDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data_path: str,
        db_path: str|List[str]|None,
        db_modification_script_path: str,
        timeout: int,
        model_name: str,
        split: Literal["train", "test"] = "train",
        n_epochs: int = 1,
        num_data: int = -1,
        use_convo_prefix: bool = True,
        max_output_tokens_per_turn: int = 3072,
        max_input_tokens: int = 32768,
        max_turns: int = 5,
        dump_path: str | None = None,
        sql_engine: str = "SQLite",
        use_verieql: bool = False,
        use_evidence_supervision_prompt: bool = False,
        use_evidence_supervision_penalty: bool = False,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if isinstance(data_path, list):
            # curriculum learning setup
            stage_1_data = cast(Dataset, load_dataset("parquet", data_files=data_path[0], keep_in_memory=True)["train"])
            stage_2_data = cast(Dataset, load_dataset("parquet", data_files=data_path[1], keep_in_memory=True)["train"])
            stage_3_data = cast(Dataset, load_dataset("parquet", data_files=data_path[2], keep_in_memory=True)["train"])
            stage_4_data = cast(Dataset, load_dataset("parquet", data_files=data_path[3], keep_in_memory=True)["train"])
            # stage 1: 2 epochs
            stage_1_data = concatenate_datasets([stage_1_data for _ in range(2)])
            print(f"Stage 1 data length after 2 epochs: {len(stage_1_data)}")
            # stage 2: 10 epochs
            stage_2_data = concatenate_datasets([stage_2_data for _ in range(10)])
            print(f"Stage 2 data length after 10 epochs: {len(stage_2_data)}")
            # stage 3: 12 epochs
            stage_3_data = concatenate_datasets([stage_3_data for _ in range(12)])
            print(f"Stage 3 data length after 12 epochs: {len(stage_3_data)}")
            # stage 4: 14 epochs
            stage_4_data = concatenate_datasets([stage_4_data for _ in range(14)])
            print(f"Stage 4 data length after 14 epochs: {len(stage_4_data)}")
            # merge all stages
            self.ds = concatenate_datasets([stage_1_data, stage_2_data, stage_3_data, stage_4_data])
            self.group_sizes = [8 for _ in range(len(stage_1_data))] + [16 for _ in range(len(stage_2_data))] + [32 for _ in range(len(stage_3_data))] + [64 for _ in range(len(stage_4_data))]

        elif isinstance(data_path, str):
            # regular setup
            self.ds = cast(Dataset, load_dataset("parquet", data_files=data_path, keep_in_memory=True)["train"])
            if split == "train":
                self.ds = self.ds.shuffle(seed=0)
                if num_data > 0:
                    self.ds = self.ds.select(range(num_data))
                if n_epochs > 1:
                    self.ds = concatenate_datasets([self.ds for _ in range(n_epochs)])
            if split == "test":
                all_ds = []
                for i in range(n_epochs):
                    ds = deepcopy(self.ds)
                    # append i to data source
                    ds = ds.map(lambda x: {"data_source": f"{x['data_source']}_{i}"})
                    all_ds.append(ds)
                self.ds = concatenate_datasets(all_ds)
                print(f"Test dataset length: {len(self.ds)}")
            self.group_sizes = [group_size for _ in range(len(self.ds))]
        else:
            raise ValueError("db_path must be a string or a list of strings")
        
        

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.db_path = db_path
        self.timeout = timeout
        self.db_modification_script_path = db_modification_script_path
        self.dump_path = dump_path
        self.use_convo_prefix = use_convo_prefix
        self.max_output_tokens_per_turn = max_output_tokens_per_turn
        self.max_input_tokens = max_input_tokens
        self.model_name = model_name
        self.max_turns = max_turns
        self.sql_engine = sql_engine
        # VeriEQL is a training-only reward shaping; never applied on the test split.
        self.use_verieql = use_verieql if split != "test" else False
        self.use_evidence_supervision_prompt = use_evidence_supervision_prompt
        self.use_evidence_supervision_penalty = use_evidence_supervision_penalty

    @classmethod
    def question_suffix(cls) -> str:
        return ""

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        group_sizes = self.group_sizes[batch_start: batch_end]
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row, group_size in zip(self.ds.select(range(batch_start, batch_end)), group_sizes)
            if (builder := self._make_env_group_builder(row, group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def set_dump_path(self, dump_path: str) -> None:
        self.dump_path = dump_path

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem_id = f"{x['data_source']}_{x['question_id']}"
        problem = x["prompt"][1]["content"]
        external_knowledge = x.get("external_knowledge", None)
        answer = x["reward_spec"]["ground_truth"]
        grading_method = x["reward_spec"].get("grading_method", "multiset")
        dataset_name = x["data_source"]
        db_id = x["db_id"]
        if self.sql_engine == "SQLite":
            db_file = f"{self.db_path}/{db_id}/{db_id}.sqlite"
        elif self.sql_engine == "Snowflake":
            db_file = db_id
        if os.path.exists(f"{self.db_modification_script_path}/{x['question_id']}.sql"):
            db_modification_script = f"{self.db_modification_script_path}/{x['question_id']}.sql"
        else:
            db_modification_script = None
        if not (problem and answer) and self.sql_engine == "SQLite" and not x['question_id'].startswith("local"):
            return None
        elif not (problem) and self.sql_engine == "Snowflake":
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                SQLEnv, problem_id, problem, external_knowledge, answer, grading_method, self.renderer, db_file, self.timeout, db_modification_script, self.dump_path, self.use_convo_prefix, self.max_output_tokens_per_turn, self.max_input_tokens, self.model_name, self.max_turns, self.sql_engine,
                use_verieql=self.use_verieql,
                use_evidence_supervision_prompt=self.use_evidence_supervision_prompt,
                use_evidence_supervision_penalty=self.use_evidence_supervision_penalty,
            ),
            num_envs=group_size,
            dataset_name=dataset_name,
        )


@chz.chz
class BIRDDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    renderer_name: str
    train_group_size: int
    base_url: str | None = None
    model_name: str
    train_data_path: str | None
    test_data_path: str | None
    db_modification_script_path: str | None
    db_path: str | None
    timeout: int = 60
    add_noise: str | None = None
    n_epochs: int = 1
    num_data: int = -1
    use_convo_prefix: bool = True
    max_output_tokens_per_turn: int = 3072
    max_input_tokens: int = 32768
    max_turns: int = 5
    curriculum_learning: bool = False
    sql_engine: str = "SQLite"
    use_verieql: bool = False
    use_evidence_supervision_prompt: bool = False
    use_evidence_supervision_penalty: bool = False

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        sql_renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        if self.curriculum_learning:
            train_data_path = [
                f"{self.train_data_path}/stage_1.parquet",
                f"{self.train_data_path}/stage_2.parquet",
                f"{self.train_data_path}/stage_3.parquet",
                f"{self.train_data_path}/stage_4.parquet",
            ]
        else:
            train_data_path = self.train_data_path
        if "db" in self.add_noise:
            assert "/databases" in self.db_path, "db_path must contain /databases if db noise is used"
            db_path = self.db_path.replace("/databases", "/original_databases")
        else:
            db_path = self.db_path

        test_data_path = self.test_data_path

        # log information about datasets
        logger.info(f"Training data path: {train_data_path}")
        logger.info(f"Test data path: {test_data_path}")
        logger.info(f"Database path: {db_path}")

        training_dataset = BIRDDataset(
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            renderer=sql_renderer,
            data_path=train_data_path,
            db_modification_script_path=self.db_modification_script_path,
            timeout=self.timeout,
            db_path=db_path,
            split="train",
            n_epochs=self.n_epochs,
            num_data=self.num_data,
            use_convo_prefix=self.use_convo_prefix,
            max_output_tokens_per_turn=self.max_output_tokens_per_turn,
            max_input_tokens=self.max_input_tokens,
            max_turns=self.max_turns,
            model_name=self.model_name,
            sql_engine=self.sql_engine,
            use_verieql=self.use_verieql,
            use_evidence_supervision_prompt=self.use_evidence_supervision_prompt,
            use_evidence_supervision_penalty=self.use_evidence_supervision_penalty,
        )
        test_dataset = BIRDDataset(
            batch_size=self.batch_size,
            group_size=1,
            renderer=sql_renderer,
            data_path=test_data_path,
            db_modification_script_path=None,
            timeout=self.timeout,
            db_path=self.db_path,
            split="test",
            n_epochs=1,
            use_convo_prefix=self.use_convo_prefix,
            max_output_tokens_per_turn=self.max_output_tokens_per_turn,
            max_input_tokens=self.max_input_tokens,
            max_turns=self.max_turns,
            model_name=self.model_name,
            sql_engine=self.sql_engine,
            use_verieql=False,
            use_evidence_supervision_prompt=self.use_evidence_supervision_prompt,
            use_evidence_supervision_penalty=False,
        )
        return training_dataset, test_dataset
