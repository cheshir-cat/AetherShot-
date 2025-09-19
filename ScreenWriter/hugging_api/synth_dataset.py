# synth_dataset.py
from __future__ import annotations
import os, json, random
from typing import List, Dict, Any
from openai import OpenAI
import instructor

from src.pydantic_schemas import SceneSchema
from src.messages import MessagesBuilder
from cache.request_saver import cache_llm_call, DiskCache, SemanticCache
from cache.postprocessor import normalize_scene

BASE_URL = "https://router.huggingface.co/v1"
API_KEY  = "hf_..."

raw_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

disk = DiskCache("./.cache/llm_cache.sqlite")
try:
    semantic = SemanticCache("./.cache/chroma", "scene_cache")
except Exception:
    semantic = None

@cache_llm_call(
    disk_cache=disk,
    semantic_cache=semantic,
    response_to_str=lambda obj: obj.model_dump_json(ensure_ascii=False),
    response_from_str=lambda s: SceneSchema.model_validate_json(s),
)
def request_scene(model: str, messages: List[Dict[str, str]], **kwargs) -> SceneSchema:
    return client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=SceneSchema,
        temperature=kwargs.get("temperature", 0.5),
        max_tokens=kwargs.get("max_tokens", 600),
    )

PROMPT_SEEDS = [
    "Ночной разговор у костра в сосновом лесу; тревожная атмосфера; лёгкий дождь.",
    "Первая встреча астронавта и робота на Луне; низкая гравитация; безмолвие.",
    "Погоня по крышам мегаполиса при неоновых вывесках; синевато-оранжевый свет.",
    "Тихий диалог в пустом музейном зале на заре; холодный свет.",
    "Сцена признания на заброшенном пирсе во время тумана; звуки воды.",
    "Интервью в маленькой кофейне; солнечный зайчик и шум улицы.",
]

def build_messages(user_text: str) -> List[Dict[str, str]]:
    return (
        MessagesBuilder()
        .system_for_schema(SceneSchema, full_json_schema=False)
        .user(user_text)
        .build()
    )

def generate_dataset(
    model: str = os.getenv("MODEL", "deepseek-ai/DeepSeek-V3-0324"),
    n: int = 1000,
    out_path: str = "./data/synthetic_scenes.jsonl",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            seed = random.choice(PROMPT_SEEDS)
            # лёгкая аугментация промпта
            aug = random.choice([
                "", " Добавь атмосферу напряжения.", " Сделай лаконичные диалоги.",
                " Упор на пластику движения камеры.", " Используй минимум цветов."
            ])
            prompt = f"{seed}{aug}".strip()
            try:
                scene = request_scene(model, build_messages(prompt))
                scene = normalize_scene(scene)  # пост-обработка
                rec = {
                    "prompt": prompt,
                    "response": scene.model_dump(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
                if cnt % 50 == 0:
                    print(f"[ok] {cnt}/{n}")
            except Exception as e:
                print(f"[skip] {i}: {e}")
                continue
    print(f"Done: {cnt} rows to {out_path}")

if __name__ == "__main__":
    generate_dataset()
