from enum import Enum
from typing import List, Optional
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
    stage: SPINStage = pydantic.Field(
        description="Текущий этап процесса продаж SPIN или 'Обработка возражений', если обнаружено возражение клиента"
    )
    detected_objections: List[str] = pydantic.Field(
        default=[],
        description="Список конкретных возражений клиента (например: Цена / Бюджет, Конкуренты, Сроки)"
    )
    suggested_scripts: List[str] = pydantic.Field(
        default=[],
        description="Рекомендуемый скрипт или фраза ответа для менеджера по продажам"
    )
    next_questions: List[str] = pydantic.Field(
        default=[],
        description="1-2 сильных вопроса для продолжения диалога и продвижения сделки"
    )
    confidence: float = pydantic.Field(
        default=1.0,
        description="Коэффициент уверенности классификации (от 0.0 до 1.0)"
    )
    customer_transcript: Optional[str] = pydantic.Field(
        default=None,
        description="Точная дословная расшифровка присланной аудиореплики на русском языке"
    )
    speaker_role: Optional[SpeakerRole] = pydantic.Field(
        default=SpeakerRole.CUSTOMER,
        description="Кто говорил в присланном аудиофайле: 'Representative' (Менеджер) или 'Customer' (Клиент)"
    )

import io
import wave

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, sample_width: int = 2) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()

class SalesAgent:
    def __init__(self):
        # Configure the Google Antigravity Agent
        self.agent_config = LocalAgentConfig(
            model=config.MODEL_NAME,
            system_instructions=config.SYSTEM_PROMPT,
            response_schema=SalesHint,
            temperature=config.TEMPERATURE
        )

    async def get_hint(self, conversation_transcript: str, new_audio_bytes: Optional[bytes] = None) -> Optional[dict]:
        """
        Sends the current conversation transcript and optional new audio bytes
        to the Antigravity Agent, returning the structured SalesHint.
        Supports MOCK_MODE to preserve API quota during local testing.
        """
        import asyncio
        if config.MOCK_MODE or not new_audio_bytes:
            # Fallback to text matching if in mock mode or no audio was provided
            if config.MOCK_MODE:
                await asyncio.sleep(0.5)  # Simulate network latency
            transcript_lower = conversation_transcript.lower()
            
            # Match objections (Price)
            if "дорог" in transcript_lower or "бюджет" in transcript_lower or "цена" in transcript_lower or "expensive" in transcript_lower:
                return {
                    "stage": "Обработка возражений",
                    "detected_objections": ["Цена / Бюджет"],
                    "suggested_scripts": [
                        "Я понимаю, что бюджет критически важен. Если наш инструмент сэкономит каждому менеджеру 5 часов в неделю, как быстро это окупит стоимость лицензии?"
                    ],
                    "next_questions": [
                        "Каковы будут потери компании, если оставить ручной ввод еще на один квартал?",
                        "Бюджет полностью заморожен, или есть возможность провести пилотный запуск?"
                    ],
                    "confidence": 0.95,
                    "customer_transcript": "Но ваше решение, скорее всего, обойдется нам слишком дорого.",
                    "speaker_role": "Customer"
                }
            # Match Implication stage
            elif "цел" in transcript_lower or "влияет" in transcript_lower or "потер" in transcript_lower or "target" in transcript_lower:
                return {
                    "stage": "Извлечение",
                    "detected_objections": [],
                    "suggested_scripts": [
                        "Это логично. Если эти 5 часов в неделю на каждого менеджера тратятся впустую, как это отражается на выполнении общего плана продаж?"
                    ],
                    "next_questions": [
                        "Что бы значило для команды освобождение этих 5 часов для активных продаж?",
                        "Какова главная цель вашего коммерческого директора на этот квартал?"
                    ],
                    "confidence": 0.88,
                    "customer_transcript": "Каждый менеджер теряет около 5 часов в неделю.",
                    "speaker_role": "Customer"
                }
            # Match Problem stage
            elif "excel" in transcript_lower or "бутылочн" in transcript_lower or "ручн" in transcript_lower or "crm" in transcript_lower:
                return {
                    "stage": "Проблема",
                    "detected_objections": [],
                    "suggested_scripts": [
                        "Работа с таблицами Excel и ручной ввод данных — классическое узкое место. Сколько времени менеджеры тратят на рутину еженедельно?"
                    ],
                    "next_questions": [
                        "Влияет ли ручное заполнение CRM на скорость обработки новых лидов?",
                        "Как сами менеджеры относятся к необходимости вручную логировать каждый звонок?"
                    ],
                    "confidence": 0.92,
                    "customer_transcript": "Честно говоря, мы ведем всё в таблицах Excel и вручную переносим в CRM. Это наше главное узкое место.",
                    "speaker_role": "Customer"
                }
            # Default to Situation stage
            else:
                return {
                    "stage": "Ситуация",
                    "detected_objections": [],
                    "suggested_scripts": [
                        "Соберите базовые факты о текущем процессе продаж клиента."
                    ],
                    "next_questions": [
                        "Какие инструменты вы сейчас используете для фиксации звонков?",
                        "Сколько менеджеров работает в вашем отделе продаж?"
                    ],
                    "confidence": 0.90,
                    "customer_transcript": "Приветствую, всё хорошо. Мы сейчас ищем способы повысить эффективность работы нашего отдела продаж.",
                    "speaker_role": "Customer"
                }

        # Multimodal request: Send conversation transcript + raw audio wav file to Gemini
        try:
            wav_bytes = pcm_to_wav(new_audio_bytes)
            audio_part = t.from_bytes(data=wav_bytes, mime_type="audio/wav")
            
            prompt = (
                "Вы анализируете этот разговор в реальном времени. Вы обязаны:\n"
                "1. Прослушать прикрепленный аудиофайл и записать точную дословную расшифровку сказанной фразы в поле `customer_transcript`.\n"
                "2. Определить, кто говорил на записи: 'Representative' (Менеджер) или 'Customer' (Клиент). Записать роль в поле `speaker_role`.\n"
                "3. На основе распознанной фразы и предыдущей истории диалога определить текущий этап SPIN-продаж или возражение и сгенерировать подсказки по SPIN.\n\n"
                "Предыдущая история диалога:\n"
                "<transcript>\n"
                f"{conversation_transcript}\n"
                "</transcript>"
            )
            
            async with Agent(config=self.agent_config) as agent:
                response = await agent.chat([prompt, audio_part])
                data = await response.structured_output()
                # Ensure the fields are mapped as dictionary
                if isinstance(data, pydantic.BaseModel):
                    return data.model_dump()
                return data
        except Exception as e:
            print(f"[SalesAgent] Multimodal error: {e}")
            return None
