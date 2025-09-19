from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Annotated, Set


system_prompt = """
Ты — сценарист-визуализатор. Сначала выполни ВНУТРЕННИЙ краткий анализ (scratchpad) — не выводи его пользователю.
Верни ТОЛЬКО валидный JSON по схеме ниже. Без префиксов/суффиксов/объяснений/markdown.
Все поля-списки (кроме custom_terms) заполняй ТОЛЬКО из шпаргалки. Формат токенов: lower_snake_case.
Максимум 3 термина на любое поле-список; если не применимо — верни [].
Если чего-то нет в шпаргалке — используй ближайший термин из списка или помести в custom_terms.
"""

Lighting = Literal[
    "chiaroscuro","high_key","low_key","rim_light","backlight","soft_light","hard_light",
    "neon","volumetric","golden_hour","blue_hour","silhouette"
]
CameraMovement = Literal[
    "dolly_in","dolly_out","dolly_zoom","push_in","pull_out","crane_up","crane_down",
    "steadicam","handheld","whip_pan","pan_left","pan_right","tilt_up","tilt_down",
    "tracking","orbit","rack_focus"
]
ShotType = Literal[
    "extreme_close_up","close_up","medium_close_up","medium_shot","cowboy_shot",
    "wide_shot","establishing","over_the_shoulder","point_of_view","two_shot","insert","cutaway"
]
Framing = Literal[
    "rule_of_thirds","leading_lines","symmetry","centered","triangular","negative_space",
    "frame_within_frame","diagonals"
]
Lens = Literal[
    "wide_24mm","28mm","35mm","50mm","85mm","135mm","macro","anamorphic_2x","fisheye"
]
Focus = Literal[
    "deep_focus","shallow_depth_of_field","rack_focus","split_diopter"
]
Palette = Literal[
    "monochrome","complementary","analogous","teal_and_orange","duotone","muted","vivid"
]
Aspect = Literal["2.39_1","1.85_1","1.78_1","1.33_1","1_1","9_16"]
TimeOfDay = Literal["dawn","morning","noon","golden_hour","blue_hour","night"]
Mood = Literal["melancholic","tense","uplifting","eerie","romantic","contemplative","epic"]
Grading = Literal[
    "bleach_bypass","cross_process","cool_tint","warm_tint","high_contrast","low_contrast"
]
CustomTerm = Annotated[str, Field(pattern=r"^[a-z0-9_]{2,64}$")]


class Dialogue(BaseModel):
    character: str
    line: str

# todo расширить
class SceneSchema(BaseModel):
    title: str
    characters: List[str]
    plot: str
    dialogues: List[Dialogue]

    lighting: List[Lighting]
    camera_movement: List[CameraMovement]
    shot_type: List[ShotType]
    framing_composition: List[Framing]
    lens: List[Lens]
    focus_technique: List[Focus]
    color_palette: List[Palette]
    aspect_ratio: Aspect
    time_of_day: TimeOfDay
    mood: Mood
    color_grading: Grading

    custom_terms: List[CustomTerm] = Field(
        default_factory=list,
        max_length=10,
        description="Редкие вне-списка термины, lower_snake_case"
    )

    # 1) нормализация custom_terms (lower_snake_case, дедуп, обрезка до 10)
    @field_validator("custom_terms", mode="before")
    @classmethod
    def _normalize_custom_terms(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            v = [v]

        def to_snake(s: str) -> str:
            s = str(s).strip().replace("-", "_").lower()
            parts = [p for p in s.split("_") if p]
            return "_".join(parts)

        seen: Set[str] = set()
        out: List[str] = []
        for item in v:
            s = to_snake(item)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:10]

    # 2) после-валидатор: удаляем из custom_terms всё, что уже указано в основных полях
    @model_validator(mode="after")
    def _dedupe_vs_primary_fields(self):
        used: Set[str] = set()

        list_fields = [
            "lighting", "camera_movement", "shot_type",
            "framing_composition", "lens", "focus_technique",
            "color_palette",
        ]
        for fname in list_fields:
            for val in getattr(self, fname, []) or []:
                if isinstance(val, str):
                    used.add(val)

        scalar_fields = ["aspect_ratio", "time_of_day", "mood", "color_grading"]
        for fname in scalar_fields:
            val = getattr(self, fname, None)
            if isinstance(val, str):
                used.add(val)

        # фильтрация custom_terms: убираем пересечения и дубли, соблюдаем лимит 10
        filtered: List[str] = []
        seen: Set[str] = set()
        for t in self.custom_terms:
            if t in used:
                continue
            if t in seen:
                continue
            seen.add(t)
            filtered.append(t)

        self.custom_terms = filtered[:10]
        return self