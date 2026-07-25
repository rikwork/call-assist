import asyncio
from enum import Enum
import json
import os
from typing import List, Optional, AsyncGenerator, Dict, Any

import httpx
import pydantic
import google.antigravity.types as t
from google.antigravity import Agent, LocalAgentConfig

import config


class SPINStage(str, Enum):
    SITUATION = "Ситуация"
    PROBLEM = "Проблема"
    IMPLICATION = "Извлечение"
    NEED_PAYOFF = "Направляющие вопросы"
    OBJECTION = "Обработка возражений"


class SpeakerRole(str, Enum):
    REPRESENTATIVE = "Representative"
    CUSTOMER = "Customer"


class SalesHint(pydantic.BaseModel):
    suggested_scripts: List[str] = pydantic.Field(
        default=[],
        description="Рекомендуемый скрипт или фраза ответа для менеджера по продажам"
    )
    next_questions: List[str] = pydantic.Field(
        default=[],
        description="1-2 сильных вопроса для продолжения диалога и продвижения сделки"
    )
    customer_transcript: str = pydantic.Field(
        default="",
        description="ОБЯЗАТЕЛЬНАЯ точная дословная расшифровка произнесенной аудиореплики на русском языке (если тишина, указывать '')"
    )
    speaker_role: SpeakerRole = pydantic.Field(
        default=SpeakerRole.REPRESENTATIVE,
        description="Кто говорил в присланном аудиофайле: 'Representative' (Менеджер) или 'Customer' (Клиент)"
    )
    stage: SPINStage = pydantic.Field(
        default=SPINStage.SITUATION,
        description="Текущий этап процесса продаж SPIN или 'Обработка возражений', если обнаружено возражение клиента"
    )
    detected_objections: List[str] = pydantic.Field(
        default=[],
        description="Список конкретных возражений клиента (например: Цена / Бюджет, Конкуренты, Сроки)"
    )
    confidence: float = pydantic.Field(
        default=1.0,
        description="Коэффициент уверенности классификации (от 0.0 до 1.0)"
    )


class SalesAgent:
    def __init__(self):
        # Configure the Google Antigravity Agent
        self.agent_config = LocalAgentConfig(
            model=config.MODEL_NAME,
            system_instructions=config.SYSTEM_PROMPT,
            response_schema=SalesHint,
            temperature=config.TEMPERATURE
        )

    async def _mock_hint(self, conversation_transcript: str) -> dict:
        transcript_lower = conversation_transcript.lower()
        if "кредит" in transcript_lower or "лимит" in transcript_lower or "карт" in transcript_lower:
            return {
                "stage": "Направляющие вопросы",
                "detected_objections": [],
                "suggested_scripts": [
                    "Никита, вместе с дебетовой картой мы можем сразу одобрить кредитную карту с 365 днями без процентов — она бесплатная и служит идеальной финансовой подушкой."
                ],
                "next_questions": [
                    "Подсказать, как зафиксировать одобренный лимит прямо сейчас?",
                    "Пользуетесь ли вы кредитной картой для выгодного кэшбэка?"
                ],
                "confidence": 0.95,
                "customer_transcript": "А какие условия по вашей кредитной карте?",
                "speaker_role": "Customer"
            }
        elif "инвест" in transcript_lower or "брок" in transcript_lower or "акци" in transcript_lower:
            return {
                "stage": "Направляющие вопросы",
                "detected_objections": [],
                "suggested_scripts": [
                    "Отлично! Также сейчас для клиентов с дебетовой картой действует акция: при открытии бесплатного брокерского счета Альфа-Инвестиций вы получаете пакет акций в подарок."
                ],
                "next_questions": [
                    "Хотите подключим брокерский счет за 1 минуту прямо к приложению?",
                    "Рассматривали ли вы вариант получать доход от свободного остатка на счете?"
                ],
                "confidence": 0.92,
                "customer_transcript": "Инвестициями я тоже немного интересовался.",
                "speaker_role": "Customer"
            }
        elif "не нужн" in transcript_lower or "дорог" in transcript_lower or "потом" in transcript_lower:
            return {
                "stage": "Обработка возражений",
                "detected_objections": ["Не нужна кредитная карта / Доп. продукты"],
                "suggested_scripts": [
                    "Никита, я вас понимаю. Карта полностью бесплатная в обслуживании — её можно просто держать про запас без каких-либо расходов."
                ],
                "next_questions": [
                    "Бывают ли ситуации, когда важно иметь под рукой средства до зарплаты?",
                    "Если карта бесплатная, удобнее активировать ее сейчас вместе с курьером?"
                ],
                "confidence": 0.90,
                "customer_transcript": "Спасибо, но мне пока нужна только дебетовая карта.",
                "speaker_role": "Customer"
            }
        else:
            return {
                "stage": "Ситуация",
                "detected_objections": [],
                "suggested_scripts": [
                    "Никита, здравствуйте! Вижу вашу заявку на дебетовую карту Альфа-Банка. Давайте подтвердим доставку и подберем выгодные опции."
                ],
                "next_questions": [
                    "Удобно ли вам получить дебетовую карту завтра курьером?",
                    "Подсказать, как зафиксировать одобренную кредитную карту с 365 днями без % и получить акции в подарок?"
                ],
                "confidence": 0.95,
                "customer_transcript": "Здравствуйте! Да, я оставлял заявку на дебетовую карту.",
                "speaker_role": "Customer"
            }

    async def get_hint_stream(
        self,
        conversation_transcript: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Pure async HTTP SSE stream using httpx.AsyncClient for zero-latency token delivery.
        Text-only input for maximum speed (~300ms latency).
        """
        if config.MOCK_MODE:
            await asyncio.sleep(0.3)
            hint_dict = await self._mock_hint(conversation_transcript)
            mock_json = json.dumps(hint_dict, ensure_ascii=False)
            yield {"type": "chunk", "text": mock_json, "full_text": mock_json}
            yield {"type": "final", "data": hint_dict}
            return

        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("[SalesAgent] GEMINI_API_KEY is not set.")
                return

            prompt = (
                "4. Сгенерируйте подсказки SalesHint по SPIN и допродажам.\n\n"
                "История предыдущего диалога (для контекста):\n"
                "<context_transcript>\n"
                f"{conversation_transcript}\n"
                "</context_transcript>"
            )

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.MODEL_NAME}:streamGenerateContent?alt=sse&key={api_key}"
            
            payload = {
                "systemInstruction": {
                    "parts": [{"text": config.SYSTEM_PROMPT}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": config.TEMPERATURE,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "suggested_scripts": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                                "description": "Рекомендуемый скрипт или фраза ответа для менеджера по продажам"
                            },
                            "next_questions": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                                "description": "1-2 сильных вопроса для продолжения диалога и продвижения сделки"
                            },
                            "customer_transcript": {
                                "type": "STRING",
                                "description": "ОБЯЗАТЕЛЬНАЯ точная дословная расшифровка произнесенной аудиореплики на русском языке"
                            },
                            "speaker_role": {
                                "type": "STRING",
                                "enum": ["Representative", "Customer"],
                                "description": "Кто говорил в присланном аудиофайле"
                            },
                            "stage": {
                                "type": "STRING",
                                "enum": ["Ситуация", "Проблема", "Извлечение", "Направляющие вопросы", "Обработка возражений"],
                                "description": "Текущий этап процесса продаж SPIN"
                            },
                            "detected_objections": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                                "description": "Список конкретных возражений клиента"
                            },
                            "confidence": {
                                "type": "NUMBER",
                                "description": "Коэффициент уверенности классификации"
                            }
                        },
                        "required": ["suggested_scripts", "next_questions", "customer_transcript", "speaker_role", "stage"]
                    }
                }
            }

            full_text = ""
            async with httpx.AsyncClient(timeout=10.0) as http_client:
                async with http_client.stream("POST", url, json=payload) as response:
                    if response.status_code == 429:
                        print("[SalesAgent] Async SSE HTTP 429 Too Many Requests hit.")
                        yield {"type": "error", "error_code": 429, "message": "Слишком много запросов"}
                        return
                    elif response.status_code != 200:
                        err_body = await response.aread()
                        print(f"[SalesAgent] Async SSE HTTP error {response.status_code}: {err_body.decode('utf-8', errors='ignore')}")
                        return

                    async for line in response.aiter_lines():
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            json_data_str = line_str[6:].strip()
                            try:
                                chunk_json = json.loads(json_data_str)
                                candidates = chunk_json.get("candidates", [])
                                if candidates:
                                    parts_res = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts_res:
                                        part_text = part.get("text", "")
                                        if part_text:
                                            full_text += part_text
                                            yield {"type": "chunk", "text": part_text, "full_text": full_text}
                            except Exception:
                                pass

            if full_text:
                try:
                    data = json.loads(full_text)
                    yield {"type": "final", "data": data}
                except Exception as e:
                    print(f"[SalesAgent] Stream JSON parse error: {e}")
        except Exception as e:
            print(f"[SalesAgent] Async SSE streaming error: {e}")

    async def get_hint(self, conversation_transcript: str) -> Optional[dict]:
        """
        Sends the current conversation transcript to the Agent, returning the structured SalesHint.
        """
        final_data = None
        async for item in self.get_hint_stream(conversation_transcript):
            if item["type"] == "final":
                final_data = item["data"]
        return final_data
