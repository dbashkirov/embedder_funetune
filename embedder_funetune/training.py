"""Fine-tuning utilities for embedding models on generated QA datasets."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer, set_seed

logger = logging.getLogger(__name__)

LossFn = Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]


@dataclass
class PairDatasetConfig:
    """Configuration describing how question-answer pairs are stored."""

    question_field: str = "question"
    answer_field: str = "answer"
    metadata_field: str = "metadata"
    max_length: int = 512


@dataclass
class FineTuningConfig:
    """Training configuration for :class:`EmbedderFineTuner`."""

    model_name_or_path: str
    dataset_path: Path
    output_dir: Path
    train_adapter: bool = False
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 0
    logging_steps: int = 10
    save_steps: int = 100
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "no"
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    save_total_limit: Optional[int] = 1
    validation_split: float = 0.0
    seed: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    report_to: Tuple[str, ...] = ("wandb",)
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class QuestionAnswerPairDataset(TorchDataset):
    """PyTorch dataset storing tokenized QA pairs."""

    def __init__(self, records: Iterable[Dict[str, str]], tokenizer, config: PairDatasetConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.samples: List[Dict[str, List[int]]] = []
        for record in records:
            question = record.get(config.question_field, "")
            answer = record.get(config.answer_field, "")
            if not question or not answer:
                logger.debug("Skipping record missing question or answer: %s", record)
                continue
            question_enc = tokenizer(
                question,
                truncation=True,
                max_length=config.max_length,
                padding=False,
            )
            answer_enc = tokenizer(
                answer,
                truncation=True,
                max_length=config.max_length,
                padding=False,
            )
            self.samples.append(
                {
                    "query_input_ids": question_enc["input_ids"],
                    "query_attention_mask": question_enc["attention_mask"],
                    "answer_input_ids": answer_enc["input_ids"],
                    "answer_attention_mask": answer_enc["attention_mask"],
                }
            )

        if not self.samples:
            raise RuntimeError("Dataset contains no valid question-answer pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


class PairDataCollator:
    """Pad query and answer sequences independently before batching."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        query_features = [
            {"input_ids": f["query_input_ids"], "attention_mask": f["query_attention_mask"]}
            for f in features
        ]
        answer_features = [
            {"input_ids": f["answer_input_ids"], "attention_mask": f["answer_attention_mask"]}
            for f in features
        ]
        query_batch = self.tokenizer.pad(query_features, return_tensors="pt")
        answer_batch = self.tokenizer.pad(answer_features, return_tensors="pt")
        batch = {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "answer_input_ids": answer_batch["input_ids"],
            "answer_attention_mask": answer_batch["attention_mask"],
        }
        return batch


class EmbedderTrainer(Trainer):
    """Custom Trainer computing loss from embedding similarity."""

    def __init__(self, *args, loss_fn: Optional[LossFn] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or default_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        query_inputs = {
            key.replace("query_", ""): value for key, value in inputs.items() if key.startswith("query_")
        }
        answer_inputs = {
            key.replace("answer_", ""): value for key, value in inputs.items() if key.startswith("answer_")
        }
        query_outputs = model(**query_inputs, output_hidden_states=True, return_dict=True)
        answer_outputs = model(**answer_inputs, output_hidden_states=True, return_dict=True)
        query_embeddings = pool_embeddings(query_outputs, query_inputs["attention_mask"])
        answer_embeddings = pool_embeddings(answer_outputs, answer_inputs["attention_mask"])
        loss = self.loss_fn(query_embeddings, answer_embeddings, inputs)
        if return_outputs:
            return loss, {
                "query_embeddings": query_embeddings,
                "answer_embeddings": answer_embeddings,
            }
        return loss


def pool_embeddings(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask."""

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    last_hidden_state = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def default_loss_fn(
    query_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor,
    _: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Default cosine embedding loss."""

    target = torch.ones(query_embeddings.size(0), device=query_embeddings.device)
    loss = torch.nn.functional.cosine_embedding_loss(query_embeddings, answer_embeddings, target)
    return loss


class EmbedderFineTuner:
    """End-to-end orchestration for fine-tuning embedding models."""

    def __init__(
        self,
        config: FineTuningConfig,
        dataset_config: Optional[PairDatasetConfig] = None,
        loss_fn: Optional[LossFn] = None,
    ):
        self.config = config
        self.dataset_config = dataset_config or PairDatasetConfig()
        self.loss_fn = loss_fn
        self._wandb_initialized = False

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)

        if self.config.seed is not None:
            set_seed(self.config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.config.model_name_or_path)
        self.model.config.use_cache = False
        self.model.config.return_dict = True

        if self.config.train_adapter:
            self._wrap_with_adapter()

    def _wrap_with_adapter(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Adapter training requires the `peft` package. Install it via `pip install peft`."
            ) from exc

        target_modules = self.config.lora_target_modules or infer_lora_target_modules(self.model)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("Enabled LoRA adapters for modules: %s", target_modules)

    def _load_records(self) -> List[Dict[str, str]]:
        path = Path(self.config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        records: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                item = json.loads(line)
                records.append(item)
        if not records:
            raise RuntimeError(f"Dataset {path} is empty")
        return records

    def _prepare_datasets(self) -> Tuple[TorchDataset, Optional[TorchDataset]]:
        records = self._load_records()
        if self.config.validation_split and 0.0 < self.config.validation_split < 1.0:
            hf_dataset = Dataset.from_list(records)
            split = hf_dataset.train_test_split(
                test_size=self.config.validation_split, seed=self.config.seed
            )
            train_records = list(split["train"])
            eval_records = list(split["test"])
            train_dataset = QuestionAnswerPairDataset(train_records, self.tokenizer, self.dataset_config)
            eval_dataset = QuestionAnswerPairDataset(eval_records, self.tokenizer, self.dataset_config)
            return train_dataset, eval_dataset

        dataset = QuestionAnswerPairDataset(records, self.tokenizer, self.dataset_config)
        return dataset, None

    def train(self) -> None:
        train_dataset, eval_dataset = self._prepare_datasets()
        data_collator = PairDataCollator(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            save_total_limit=self.config.save_total_limit,
            report_to=list(self.config.report_to),
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
        )

        self._init_wandb()

        trainer = EmbedderTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=self.loss_fn,
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)

        if self._wandb_initialized:
            try:
                import wandb
            except ImportError:  # pragma: no cover - defensive
                pass
            else:
                wandb.finish()

    def _init_wandb(self) -> None:
        if "wandb" not in self.config.report_to:
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("W&B logging requires the `wandb` package. Install it via `pip install wandb`.") from exc

        project = self.config.wandb_project or os.getenv("WANDB_PROJECT")
        if not project:
            raise RuntimeError("WANDB_PROJECT environment variable or wandb_project in config must be set")

        if wandb.run is not None:
            return

        wandb.init(
            project=project,
            entity=self.config.wandb_entity or os.getenv("WANDB_ENTITY"),
            name=self.config.wandb_run_name,
            config={
                "model_name_or_path": self.config.model_name_or_path,
                "learning_rate": self.config.learning_rate,
                "train_adapter": self.config.train_adapter,
                "num_train_epochs": self.config.num_train_epochs,
            },
        )
        self._wandb_initialized = True


def infer_lora_target_modules(model) -> List[str]:
    """Infer a sensible default list of modules for LoRA adaptation."""

    candidates = set()
    for name, module in model.named_modules():
        if any(key in name for key in ("q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value")):
            candidates.add(name.split(".")[-1])
    if not candidates:
        candidates.update({"dense", "fc1", "fc2"})
    return sorted(candidates)


__all__ = [
    "EmbedderFineTuner",
    "FineTuningConfig",
    "PairDatasetConfig",
    "default_loss_fn",
    "pool_embeddings",
]
