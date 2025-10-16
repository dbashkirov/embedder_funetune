"""Question generation utilities for building QA datasets from vector stores."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - defensive import
    raise ImportError(
        "The `openai` package is required to use question generation. Install it via `pip install openai`."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class QuestionGenerationConfig:
    """Configuration for :class:`QuestionGenerator`."""

    collection_name: str
    db_type: str
    host: str
    port: int
    prompt_path: Path
    output_path: Path
    model: str
    env_path: Path = Path(".env")
    text_key: str = "text"
    system_prompt: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 0.9
    max_chunks: Optional[int] = None
    batch_size: int = 64
    question_count: int = 5
    prompt_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    """A single text chunk retrieved from a vector store."""

    id: str
    text: str
    metadata: Dict[str, object]


class BaseChunkSource:
    """Abstract base class for chunk iterators."""

    def __iter__(self) -> Iterator[ChunkRecord]:  # pragma: no cover - runtime implemented by subclasses
        raise NotImplementedError


class ChromaChunkSource(BaseChunkSource):
    """Iterate over chunks stored in a ChromaDB collection."""

    def __init__(self, config: QuestionGenerationConfig):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The `chromadb` package is required for reading from ChromaDB. Install it via `pip install chromadb`."
            ) from exc

        self._collection_name = config.collection_name
        self._text_key = config.text_key
        self._batch_size = config.batch_size
        self._client = chromadb.HttpClient(host=config.host, port=config.port)
        self._collection = self._client.get_collection(config.collection_name)

    def __iter__(self) -> Iterator[ChunkRecord]:
        offset = 0
        while True:
            logger.debug("Fetching batch from Chroma starting at offset %d", offset)
            batch = self._collection.get(
                include=["ids", "documents", "metadatas"],
                limit=self._batch_size,
                offset=offset,
            )
            ids: List[str] = batch.get("ids", []) or []
            documents: List[Optional[str]] = batch.get("documents", []) or []
            metadatas: List[Optional[Dict[str, object]]] = batch.get("metadatas", []) or []

            if not ids:
                logger.debug("No more chunks found in Chroma collection %s", self._collection_name)
                break

            for idx, chunk_id in enumerate(ids):
                metadata = metadatas[idx] or {}
                document_text = documents[idx]
                text = (document_text or metadata.get(self._text_key) or "").strip()
                if not text:
                    logger.debug("Skipping chunk %s with empty text", chunk_id)
                    continue
                yield ChunkRecord(id=chunk_id, text=text, metadata=metadata)

            offset += len(ids)


class QdrantChunkSource(BaseChunkSource):
    """Iterate over chunks stored in a Qdrant collection."""

    def __init__(self, config: QuestionGenerationConfig):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The `qdrant-client` package is required for reading from Qdrant. Install it via `pip install qdrant-client`."
            ) from exc

        self._client = QdrantClient(host=config.host, port=config.port)
        self._collection_name = config.collection_name
        self._text_key = config.text_key
        self._batch_size = config.batch_size

    def __iter__(self) -> Iterator[ChunkRecord]:
        next_page = None
        while True:
            logger.debug("Scrolling Qdrant collection %s", self._collection_name)
            points, next_page = self._client.scroll(
                collection_name=self._collection_name,
                limit=self._batch_size,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                logger.debug("No more points found in Qdrant collection %s", self._collection_name)
                break

            for point in points:
                payload = point.payload or {}
                text = (payload.get(self._text_key) or "").strip()
                if not text:
                    logger.debug("Skipping point %s due to missing '%s' in payload", point.id, self._text_key)
                    continue
                metadata = dict(payload)
                yield ChunkRecord(id=str(point.id), text=text, metadata=metadata)

            if next_page is None:
                break


def _build_chunk_source(config: QuestionGenerationConfig) -> BaseChunkSource:
    db_type = config.db_type.lower()
    if db_type == "chroma":
        return ChromaChunkSource(config)
    if db_type == "qdrant":
        return QdrantChunkSource(config)
    raise ValueError(f"Unsupported db_type: {config.db_type}. Expected 'chroma' or 'qdrant'.")


class QuestionGenerator:
    """Generate question-answer pairs for documents stored in a vector database."""

    def __init__(self, config: QuestionGenerationConfig):
        self.config = config
        self._prompt_template = Path(config.prompt_path).read_text(encoding="utf-8")

        load_dotenv(config.env_path)
        api_key = os.getenv("YC_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("YC_OPENAI_API_KEY (or OPENAI_API_KEY) must be defined in the environment or .env file")
        base_url = os.getenv("YC_OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL"))

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._chunk_source = _build_chunk_source(config)

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)

    def generate(self) -> Path:
        """Generate questions for each chunk and persist them as JSONL pairs."""

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        for idx, chunk in enumerate(self._chunk_iterator(), start=1):
            logger.info("Generating questions for chunk %s", chunk.id)
            prompt = self._format_prompt(chunk)
            try:
                questions = self._call_model(prompt)
            except Exception:  # pragma: no cover - propagate but log
                logger.exception("Failed to generate questions for chunk %s", chunk.id)
                continue

            answer_text = chunk.text.strip()
            if not answer_text:
                logger.debug("Skipping chunk %s with empty answer text", chunk.id)
                continue

            for question in questions:
                question_text = question.strip()
                if not question_text:
                    logger.debug("Skipping empty question for chunk %s", chunk.id)
                    continue
                rows.append(
                    {
                        "chunk_id": chunk.id,
                        "question": question_text,
                        "answer": answer_text,
                        "metadata": chunk.metadata,
                    }
                )

        if not rows:
            raise RuntimeError("No questions were generated. Check model outputs and prompt formatting.")

        with output_path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Saved %d question-answer pairs to %s", len(rows), output_path)
        return output_path

    def _chunk_iterator(self) -> Iterable[ChunkRecord]:
        count = 0
        for chunk in self._chunk_source:
            yield chunk
            count += 1
            if self.config.max_chunks is not None and count >= self.config.max_chunks:
                break

    def _format_prompt(self, chunk: ChunkRecord) -> str:
        variables = dict(self.config.prompt_variables)
        variables.setdefault("context", chunk.text)
        variables.setdefault("metadata", json.dumps(chunk.metadata, ensure_ascii=False))
        variables.setdefault("question_count", str(self.config.question_count))
        base_prompt = self._prompt_template.format(**variables).rstrip()
        format_instructions = (
            "\n\n"
            "Ответ верни строго в формате JSON-объекта с ключом \"questions\" и списком строк вопросов. "
            "Пример: {\"questions\": [\"Вопрос 1?\", \"Вопрос 2?\"]}. "
            "Никаких дополнительных полей, комментариев или пояснений не добавляй."
        )
        return f"{base_prompt}{format_instructions}"

    def _call_model(self, prompt: str) -> List[str]:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        content = response.choices[0].message["content"].strip()
        logger.debug("Raw model output: %s", content)
        parsed = self._parse_model_output(content)
        if not parsed:
            raise ValueError("Model response did not contain any valid questions")
        return parsed

    @staticmethod
    def _parse_model_output(content: str) -> List[str]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return _parse_fallback(content)

        if isinstance(data, dict):
            # Support responses shaped like {"questions": [...]} for robustness
            if "questions" in data and isinstance(data["questions"], list):
                data = data["questions"]
            else:
                data = [data]
        if not isinstance(data, list):
            return []

        parsed_questions: List[str] = []
        for item in data:
            if isinstance(item, str):
                parsed_questions.append(item)
            elif isinstance(item, dict):
                question = item.get("question") or item.get("q")
                if isinstance(question, str):
                    parsed_questions.append(question)
        cleaned = [question.strip() for question in parsed_questions if isinstance(question, str) and question.strip()]
        return [question for question in cleaned if "?" in question]


def _parse_fallback(content: str) -> List[str]:
    """Fallback parser for simple textual outputs."""

    questions: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("question:") or lower.startswith("q:"):
            candidate = line.split(":", 1)[1].strip()
        elif line[0].isdigit() and ("." in line or ")" in line):
            delimiter = "." if "." in line else ")"
            candidate = line.split(delimiter, 1)[1].strip()
        elif line.startswith("-") or line.startswith("•"):
            candidate = line[1:].strip()
        else:
            candidate = line
        if "?" in candidate:
            questions.append(candidate)
    return questions


__all__ = ["QuestionGenerator", "QuestionGenerationConfig"]
