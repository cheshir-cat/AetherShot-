from __future__ import annotations
from typing import Iterable, Tuple, Type, List, Dict, Literal
from pydantic import BaseModel


#todo expand it
SYSTEM_TEMPLATE = """Ты — сценарист-визуализатор. Сначала выполни ВНУТРЕННИЙ краткий анализ (scratchpad) — не выводи его пользователю.
Верни ТОЛЬКО валидный JSON по схеме ниже. Без префиксов/суффиксов/объяснений/markdown.
Все поля-списки (кроме custom_terms) заполняй ТОЛЬКО из шпаргалки. Формат токенов: lower_snake_case.
Максимум 3 термина на любое поле-список; если не применимо — верни [].
Если чего-то нет в шпаргалке — используй ближайший термин из списка или помести в custom_terms.
"""


class MessagesBuilder:
    def __init__(self) -> None:
        self._messages: List[Dict[str, str]] = []

    @staticmethod
    def _schema_text(model: Type[BaseModel], full_json_schema: bool = False) -> str:
        # 1) компактная инструкция из полей (экономит токены)
        if not full_json_schema:
            fields = []
            for name, field in model.model_fields.items():
                typ = getattr(field.annotation, "__name__", str(field.annotation))
                fields.append(f'- "{name}": {typ}')
            return "{\n" + "\n".join(fields) + "\n}"
        # 2) или полноразмерный JSON Schema (дороже по токенам)
        return model.model_json_schema()

    def system_for_schema(self, model: Type[BaseModel], *, full_json_schema: bool = False) -> "MessagesBuilder":
        schema_str = self._schema_text(model, full_json_schema=full_json_schema)
        self._messages.append({"role": "system", "content": SYSTEM_TEMPLATE.format(schema=schema_str)})
        return self

    def fewshots(self, examples: Iterable[Tuple[str, str]]) -> "MessagesBuilder":
        # examples: [(user_text, assistant_json), ...]
        for u, a in examples:
            self._messages.append({"role": "user", "content": u})
            self._messages.append({"role": "assistant", "content": a})
        return self

    def history(self, turns: Iterable[Tuple[str, str]]) -> "MessagesBuilder":
        # Произвольная история: (role, content)
        for role, content in turns:
            self._messages.append({"role": role, "content": content})
        return self

    def user(self, content: str) -> "MessagesBuilder":
        self._messages.append({"role": "user", "content": content})
        return self

    def build(self) -> List[Dict[str, str]]:
        return self._messages[:]