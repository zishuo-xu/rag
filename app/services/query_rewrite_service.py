import re
from dataclasses import dataclass


@dataclass
class RewriteResult:
    original_question: str
    rewritten_question: str
    applied_rules: list[str]


class QueryRewriteService:
    def rewrite(self, question: str) -> RewriteResult:
        original = question.strip()
        rewritten = original
        applied_rules: list[str] = []

        normalized = re.sub(r"\s+", " ", rewritten)
        if normalized != rewritten:
            rewritten = normalized
            applied_rules.append("normalize_whitespace")

        rewritten_candidate, replaced = _replace_pronouns(rewritten)
        if replaced:
            rewritten = rewritten_candidate
            applied_rules.append("replace_pronouns")

        rewritten_candidate, expanded = _expand_short_questions(rewritten)
        if expanded:
            rewritten = rewritten_candidate
            applied_rules.append("expand_short_question")

        rewritten_candidate, clarified = _clarify_topic_words(rewritten)
        if clarified:
            rewritten = rewritten_candidate
            applied_rules.append("clarify_topic_words")

        rewritten_candidate, expanded = _expand_retrieval_semantics(rewritten)
        if expanded:
            rewritten = rewritten_candidate
            applied_rules.append("expand_retrieval_semantics")

        return RewriteResult(
            original_question=original,
            rewritten_question=rewritten,
            applied_rules=applied_rules,
        )


def _replace_pronouns(question: str) -> tuple[str, bool]:
    pronouns = ["它", "这家公司", "这个公司", "该公司", "这家企业", "这家厂商"]
    for pronoun in pronouns:
        if pronoun in question:
            return question.replace(pronoun, "目标公司", 1), True
    return question, False


def _expand_short_questions(question: str) -> tuple[str, bool]:
    stripped = question.strip()
    if any(phrase in stripped for phrase in ["情况如何", "是什么", "怎么样", "如何"]):
        return question, False
    if stripped.endswith(("情况", "现状", "作用", "前景", "风险", "估值", "布局", "表现", "业务")):
        return f"{stripped}如何", True
    if len(stripped) <= 12 and not stripped.endswith("？") and not stripped.endswith("?"):
        return f"{stripped}的具体情况是什么？", True
    return question, False


def _clarify_topic_words(question: str) -> tuple[str, bool]:
    replacements = {
        "前景": "发展前景",
        "机会": "增长机会",
        "风险": "主要风险",
        "走势": "发展走势",
        "发展呢": "发展情况如何",
        "怎么样": "情况如何",
    }
    rewritten = question
    changed = False
    for source, target in replacements.items():
        if source in rewritten:
            rewritten = rewritten.replace(source, target)
            changed = True
    rewritten = rewritten.replace("情况如何的具体情况是什么", "情况如何")
    rewritten = rewritten.replace("如何的具体情况是什么", "如何")
    return rewritten, changed


def _expand_retrieval_semantics(question: str) -> tuple[str, bool]:
    lowered = question.lower()
    additions: list[str] = []

    if any(marker in question for marker in ["海外", "国际市场", "全球化", "出海"]):
        additions.extend(["美国市场", "墨西哥工厂", "摩洛哥基地", "欧洲市场", "客户合作"])

    if "上市" in question and any(marker in question for marker in ["时间", "什么时候", "哪年", "何时"]):
        additions.extend(["上市时间", "上海证券交易所", "股票代码"])

    if any(marker in question for marker in ["准入试点", "上路通行试点", "试点"]) and any(
        marker in question for marker in ["关注哪些", "关注重点", "重点事项", "主要关注"]
    ):
        additions.extend(["试点关注重点", "核心事项", "安全保障", "准入要求"])

    if "客户" in lowered and "合作" not in lowered:
        additions.append("合作关系")

    deduped = [item for item in dict.fromkeys(additions) if item not in question]
    if not deduped:
        return question, False
    return f"{question} {' '.join(deduped)}", True
