import re


def derive_semantic_tags(text: str, section_title: str | None = None) -> list[str]:
    source = f"{section_title or ''}\n{text}".lower()
    tags: set[str] = set()

    tag_rules = {
        "globalization": ["海外", "国际", "全球化", "欧美市场", "国际市场"],
        "us_market": ["美国", "北美", "美墨加"],
        "mexico_factory": ["墨西哥", "美墨加协定"],
        "morocco_factory": ["摩洛哥"],
        "europe_market": ["欧洲", "欧洲市场", "沃尔沃"],
        "company_profile": ["基本情况", "公司概况", "主营业务", "高新技术企业", "上市"],
        "finance": ["财务", "营收", "营业收入", "净利润", "毛利率", "净利率", "现金流", "资产负债"],
        "valuation": ["估值", "市盈率", "市净率", "总市值", "目标价"],
        "smart_driving": ["智能驾驶", "adas", "emb", "wcbs", "线控制动", "智驾"],
        "customer": ["客户", "主机厂", "供应链", "通用", "福特", "大众", "丰田", "沃尔沃"],
    }

    for tag, keywords in tag_rules.items():
        if any(keyword.lower() in source for keyword in keywords):
            tags.add(tag)

    if section_title:
        lowered_title = section_title.lower()
        if re.search(r"客户|合作", lowered_title):
            tags.add("customer")
        if re.search(r"财务|估值", lowered_title):
            tags.add("finance")
        if re.search(r"基本情况|概况", lowered_title):
            tags.add("company_profile")

    return sorted(tags)
