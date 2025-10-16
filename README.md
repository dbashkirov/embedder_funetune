# embedder_funetune

Библиотека предназначена для генерации вопросно-ответных пар из векторных баз данных (ChromaDB или Qdrant) и последующего дообучения моделей-эмбеддеров на полученном датасете.

## Установка

```bash
pip install -e .
```

Минимально необходимы переменные окружения для доступа к OpenAI-совместимой модели в Яндекс.Облаке. Укажите их в файле `.env` или в окружении:

```env
YC_OPENAI_API_KEY=...  # ключ для API
YC_OPENAI_API_BASE=https://llm.api.cloud.yandex.net/foundationModels/v1  # кастомный base URL (опционально)
```

Для мониторинга обучения в Weights & Biases необходимо задать `WANDB_PROJECT` (и при необходимости `WANDB_ENTITY`).

## Генерация вопросно-ответных пар

```python
from pathlib import Path
from embedder_funetune import QuestionGenerationConfig, QuestionGenerator

config = QuestionGenerationConfig(
    collection_name="documents",
    db_type="chroma",  # или "qdrant"
    host="localhost",
    port=8000,
    prompt_path=Path("prompts/qa_prompt.txt"),
    output_path=Path("data/generated_qa.jsonl"),
    model="yandexgpt",
    system_prompt="Ты помогаешь писать вопросы для системы поиска информации.",
)

generator = QuestionGenerator(config)
output_file = generator.generate()
print(f"Датасет сохранён в {output_file}")
```

Файл с промптом должен содержать шаблон, в который будет подставлен текст чанка через плейсхолдер `{context}`. При необходимости можно добавить дополнительные переменные через `config.prompt_variables`.

Пример `prompts/qa_prompt.txt`:

```
Сгенерируй 3 вопроса и ответы по контексту ниже.
Контекст:
{context}

Верни результат в формате JSON-списка объектов вида {{"question": "...", "answer": "..."}}.
```

## Дообучение модели-эмбеддера

```python
from pathlib import Path
from embedder_funetune import EmbedderFineTuner, FineTuningConfig

config = FineTuningConfig(
    model_name_or_path="intfloat/multilingual-e5-base",
    dataset_path=Path("data/generated_qa.jsonl"),
    output_dir=Path("artifacts/e5-finetuned"),
    train_adapter=True,  # True — учим LoRA адаптеры, False — всю модель
    num_train_epochs=2,
    learning_rate=2e-5,
    wandb_run_name="qa-embedder"
)

finetuner = EmbedderFineTuner(config)
finetuner.train()
```

По умолчанию используется косинусная функция потерь. Её можно заменить, передав собственную функцию `loss_fn`, принимающую эмбеддинги вопросов и ответов и возвращающую скаляр потерь.

```python
import torch

def contrastive_loss(query_embeds: torch.Tensor, answer_embeds: torch.Tensor, _):
    logits = torch.matmul(query_embeds, answer_embeds.T)
    labels = torch.arange(logits.size(0), device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)

finetuner = EmbedderFineTuner(config, loss_fn=contrastive_loss)
finetuner.train()
```

## Формат датасета

Сгенерированные данные сохраняются в `JSONL` с полями:

- `chunk_id` — идентификатор чанка в векторной БД;
- `question` — сгенерированный вопрос;
- `answer` — ответ модели;
- `metadata` — оригинальные метаданные чанка.

Этот же формат используется на этапе обучения.

## Лицензия

MIT
