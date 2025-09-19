from __future__ import annotations
import re
from typing import List, Dict, Any, Iterable, Set
from ..src.pydantic_schemas import SceneSchema

SYNONYMS = {
    # camera movement
    "dolly zoom": "dolly_zoom",
    "dolly-zoom": "dolly_zoom",
    "vertigo effect": "dolly_zoom",
    "push in": "push_in",
    "pull out": "pull_out",
    "rack focus": "rack_focus",
    # lighting
    "chiarro scuro": "chiaroscuro",
    "chiaro_scuro": "chiaroscuro",
    "low key": "low_key",
    "high key": "high_key",
    "rim light": "rim_light",
    # misc
    "teal & orange": "teal_and_orange",
    "2.39:1": "2.39_1",
    "1.85:1": "1.85_1",
}

def to_snake_token(s: str) -> str:
    s = s.strip().lower()
    s = SYNONYMS.get(s, s)
    s = s.replace("&", "and").replace("-", "_").replace(":", "_").replace("/", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def norm_list(values: Iterable[str], limit: int = 3) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for v in values or []:
        token = to_snake_token(str(v))
        token = SYNONYMS.get(token, token)
        if token and token not in seen:
            seen.add(token)
            out.append(token)
        if len(out) >= limit:
            break
    return out

def normalize_scene(scene: SceneSchema) -> SceneSchema:
    data: Dict[str, Any] = scene.model_dump()
    # нормализуем списки-токены
    for key in [
        "lighting","camera_movement","shot_type","framing_composition",
        "lens","focus_technique","color_palette",
    ]:
        data[key] = norm_list(data.get(key, []), limit=3)

    # scalar-поля
    for key in ["aspect_ratio","time_of_day","mood","color_grading"]:
        val = data.get(key)
        if isinstance(val, str):
            data[key] = to_snake_token(val)

    # custom_terms: до 10, snake_case, без дублей остальных полей
    others = set()
    for key in [
        "lighting","camera_movement","shot_type","framing_composition",
        "lens","focus_technique","color_palette",
    ]:
        others.update(data.get(key, []))
    for key in ["aspect_ratio","time_of_day","mood","color_grading"]:
        v = data.get(key)
        if isinstance(v, str):
            others.add(v)

    c_terms = [to_snake_token(x) for x in (data.get("custom_terms") or [])]
    c_norm: List[str] = []
    for t in c_terms:
        if t and t not in others and t not in c_norm:
            c_norm.append(t)
        if len(c_norm) >= 10:
            break
    data["custom_terms"] = c_norm

    # повторная валидация pydantic (если schema содержит Literal-ограничения — отловит ошибки)
    return SceneSchema.model_validate(data)
