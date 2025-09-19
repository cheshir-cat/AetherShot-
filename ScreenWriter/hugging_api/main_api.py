from __future__ import annotations

from typing import List, Dict

from pydantic import ValidationError
from openai import OpenAI, APIStatusError
import instructor

from ScreenWriter.hugging_api.src.pydantic_schemas import SceneSchema
from ScreenWriter.hugging_api.src.messages import MessagesBuilder
from cache.postprocessor import normalize_scene


HF_BASE_URL = "https://router.huggingface.co/v1"
HF_TOKEN    = "hf_"
fewshot_examples = [
    (
        "Короткая сцена знакомства пилота дрона и оператора в пустыне.",
        """{
            "title": "Песок и Сигнал",
            "characters": ["Оператор", "Пилот"],
            "plot": "Оператор теряет дрон в песчаной буре и просит помощи у пилота.",
            "dialogues": [
                {"character": "Оператор", "line": "Связь пропадает... Видишь что-нибудь?"},
                {"character": "Пилот", "line": "Держу высоту. Поверну на восток, там меньше помех."}
            ]
        }"""
    )
]


raw_client = OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN)
client = instructor.from_openai(raw_client,
                                mode=instructor.Mode.JSON)

def request_scene(model: str, messages: List[Dict[str, str]]) -> SceneSchema:

    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=SceneSchema
        )
    except APIStatusError as e:
        print(f"❌ API error {e.status_code}: {e}")
    except ValidationError as ve:
        print("⚠️ JSON не прошёл валидацию схемы:\n", ve)


if __name__ == "__main__":
    msgs = (
        MessagesBuilder()
        .system_for_schema(SceneSchema, full_json_schema=False)
        .fewshots(fewshot_examples)
        .user("Сцена первой встречи астронавта и робота на Луне.") #todo expand request
        .build()
    )
    scene = request_scene("deepseek-ai/DeepSeek-V3-0324", msgs)
    print(scene.model_dump_json(indent=2))

    #scene = normalize_scene(scene)
