from dataclasses import dataclass

from openai import OpenAI

from app.core.config import get_settings


@dataclass
class LLMResult:
    answer: str
    model_name: str
    input_text: str
    output_text: str
    provider_status: str
    fallback_reason: str | None = None


class OpenAILLMProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.llm_model
        self.enabled = bool(settings.llm_api_key and settings.llm_model)
        self.base_url = settings.llm_base_url.rstrip("/")
        self.timeout_seconds = 45.0
        self.client = None
        if self.enabled:
            self.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=self.timeout_seconds,
            )
        self.prefers_chat_completions = "api.deepseek.com" in self.base_url

    def generate_answer(self, *, question: str, context_blocks: list[str]) -> LLMResult:
        if not self.enabled or not self.client:
            raise RuntimeError("llm is not configured")

        prompt = self.build_prompt(question=question, context_blocks=context_blocks)
        responses_error: str | None = None

        if not self.prefers_chat_completions:
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                output = (response.output_text or "").strip()
                if output:
                    return LLMResult(
                        answer=output,
                        model_name=self.model,
                        input_text=prompt,
                        output_text=output,
                        provider_status="external_responses",
                        fallback_reason=None,
                    )
            except Exception as exc:
                responses_error = f"responses_failed: {type(exc).__name__}: {exc}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个企业知识库问答助手，必须严格基于提供的资料回答，不要编造信息。"
                        "回答中的关键结论后尽量附上 [1] 这类引用编号，且只能使用已提供的编号。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        message = completion.choices[0].message.content if completion.choices else ""
        output = (message or "").strip()
        return LLMResult(
            answer=output,
            model_name=self.model,
            input_text=prompt,
            output_text=output,
            provider_status="external_chat_completions",
            fallback_reason=responses_error if not self.prefers_chat_completions else None,
        )

    def build_prompt(self, *, question: str, context_blocks: list[str]) -> str:
        context = "\n\n".join(context_blocks)
        return (
            "你是一个企业知识库问答助手。"
            "你必须严格根据提供的参考资料回答。"
            "如果参考资料不足以支持结论，请明确回答“知识库暂无足够依据”。"
            "请把答案控制在 120 个汉字以内。"
            "先给出一句简洁结论，再列出不超过 3 条依据要点。"
            "优先复用参考资料中的实体词和关键术语，不要自由发挥改写。"
            "每个关键结论必须能在参考资料中找到直接依据。"
            "优先使用更具体的事实依据，不要先复述结论性或投资建议性段落。"
            "如果参考资料里既有具体业务事实，又有总结性段落，优先引用具体业务事实。"
            "不要把重复片段当作多份独立证据。"
            "如果资料之间缺少足够支撑，请降低结论强度，不要过度推断。"
            "回答中的关键结论后尽量追加引用编号，例如 [1] 或 [1][2]。"
            "不要编造不存在的引用编号。"
            f"\n\n用户问题：{question}"
            f"\n\n参考资料：\n{context}"
        )
