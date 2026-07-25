import asyncio
import os
import unittest
import config
from agent import SalesAgent, SalesHint, SPINStage, SpeakerRole


class TestSalesAgent(unittest.TestCase):
    def setUp(self):
        config.MOCK_MODE = True
        self.agent = SalesAgent()

    def test_schema_instantiation(self):
        hint = SalesHint(
            suggested_scripts=["Здравствуйте!"],
            next_questions=["Удобно ли общаться?"],
            customer_transcript="Да, удобнее.",
            speaker_role=SpeakerRole.CUSTOMER,
            detected_objections=[]
        )
        self.assertEqual(hint.speaker_role, "Customer")
        self.assertEqual(hint.stage, "Ситуация")

    def test_mock_hint_retrieval(self):
        async def run_test():
            result = await self.agent.get_hint("Клиент интересуется кредитной картой")
            self.assertIsNotNone(result)
            self.assertIn("stage", result)
            self.assertIn("suggested_scripts", result)
            self.assertEqual(result["speaker_role"], "Customer")

        asyncio.run(run_test())

    def test_hint_stream_chunks(self):
        async def run_test():
            chunks = []
            final_data = None
            async for item in self.agent.get_hint_stream("Рассматриваю инвестиции"):
                if item["type"] == "chunk":
                    chunks.append(item["text"])
                elif item["type"] == "final":
                    final_data = item["data"]

            self.assertTrue(len(chunks) > 0)
            self.assertIsNotNone(final_data)
            self.assertEqual(final_data["stage"], "Направляющие вопросы")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
