#!/usr/bin/env python3
"""Generate bilingual prompt-illustration docs for ReXGroundingCT."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METADATA = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
DEFAULT_OUTPUT_DIR = Path("docs/voxtell/prompts")
DOC_DATE = "2026-07-22"
PROMPT_UNIQUE_DOC = "unique_prompts_bilingual.md"
PROMPT_INSTANCES_DOC = "prompt_instances_bilingual.md"
PROMPT_VOCAB_DOC = "vocabulary_bilingual.md"
SPLITS = ("train", "val", "test")
EXPECTED_PROMPT_COUNTS = {"train": 7687, "val": 381, "test": 582}
EXPECTED_UNIQUE_PROMPTS = 6926


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "&#124;")


def write_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"created: {DOC_DATE}\n")
        f.write(f"updated: {DOC_DATE}\n")
        f.write("status: active\n")
        f.write("---\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(md_escape(cell) for cell in row) + " |\n")


def finding_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def iter_instances(metadata: dict) -> list[dict]:
    instances = []
    for split in SPLITS:
        for entry in metadata[split]:
            findings = entry.get("findings", {})
            categories = entry.get("categories", {})
            entity_counts = entry.get("entity_counts", {})
            for finding_id in sorted(findings, key=finding_key):
                instances.append(
                    {
                        "split": split,
                        "case_name": entry["name"],
                        "finding_id": finding_id,
                        "category": categories.get(finding_id, ""),
                        "entity_count": entity_counts.get(finding_id, ""),
                        "prompt": findings[finding_id],
                    }
                )
    return instances


def phrase_pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE)


def apply_outside_parentheses(text: str, rules: list[tuple[re.Pattern, str]]) -> str:
    for pattern, replacement in rules:
        parts = re.split(r"(（[^）]*）)", text)
        for index, part in enumerate(parts):
            if part.startswith("（") and part.endswith("）"):
                continue
            part = pattern.sub(replacement, part)
            parts[index] = part
        text = "".join(parts)
    return text


def add_phrase(mapping: dict[str, str], phrase: str, zh: str, include_english: bool = True) -> None:
    mapping[phrase.lower()] = f"{zh}（{phrase.lower()}）" if include_english else zh


def build_translation_rules() -> list[tuple[re.Pattern, str]]:
    phrases: dict[str, str] = {}

    lobes = {
        "right upper lobe": "右上叶",
        "right middle lobe": "右中叶",
        "right lower lobe": "右下叶",
        "left upper lobe": "左上叶",
        "left lower lobe": "左下叶",
        "upper lobe of the right lung": "右肺上叶",
        "middle lobe of the right lung": "右肺中叶",
        "lower lobe of the right lung": "右肺下叶",
        "upper lobe of the left lung": "左肺上叶",
        "lower lobe of the left lung": "左肺下叶",
    }
    segments = {
        "apical segment": "尖段",
        "posterior segment": "后段",
        "anterior segment": "前段",
        "superior segment": "上段",
        "inferior segment": "下段",
        "medial segment": "内侧段",
        "lateral segment": "外侧段",
        "basal segment": "基底段",
        "anterobasal segment": "前基底段",
        "anterior basal segment": "前基底段",
        "posterobasal segment": "后基底段",
        "posterior basal segment": "后基底段",
        "laterobasal segment": "外侧基底段",
        "lateral basal segment": "外侧基底段",
        "mediobasal segment": "内侧基底段",
        "medial basal segment": "内侧基底段",
        "anterolateral basal segment": "前外侧基底段",
        "posterolateral segment": "后外侧段",
        "apicoposterior segment": "尖后段",
        "lingular segment": "舌段",
        "superior lingular segment": "上舌段",
        "inferior lingular segment": "下舌段",
    }
    for phrase, zh in lobes.items():
        add_phrase(phrases, phrase, zh)
    for phrase, zh in segments.items():
        add_phrase(phrases, phrase, zh)
    for lobe, lobe_zh in lobes.items():
        if " of the " in lobe:
            continue
        for segment, segment_zh in segments.items():
            add_phrase(phrases, f"{segment} of the {lobe}", f"{lobe_zh}{segment_zh}")
            add_phrase(phrases, f"{lobe} {segment}", f"{lobe_zh}{segment_zh}")

    direct_phrases = {
        "subcentimeter nonspecific pulmonary nodules": "亚厘米非特异性肺结节",
        "subcentimeter nonspecific parenchymal nodules": "亚厘米非特异性肺实质结节",
        "subcentimeter nonspecific nodules": "亚厘米非特异性结节",
        "subcentimeter non-specific nodules": "亚厘米非特异性结节",
        "subcentimeter pulmonary nodules": "亚厘米肺结节",
        "subcentimeter parenchymal nodules": "亚厘米肺实质结节",
        "subcentimeter nodules": "亚厘米结节",
        "ground glass opacities": "磨玻璃影",
        "ground-glass opacities": "磨玻璃影",
        "ground glass opacity": "磨玻璃影",
        "ground-glass opacity": "磨玻璃影",
        "ground glass densities": "磨玻璃密度影",
        "ground-glass densities": "磨玻璃密度影",
        "ground glass density": "磨玻璃密度影",
        "ground-glass density": "磨玻璃密度影",
        "ground glass appearance": "磨玻璃样表现",
        "ground-glass appearance": "磨玻璃样表现",
        "ground glass consolidation": "磨玻璃样实变",
        "ground glass consolidations": "磨玻璃样实变",
        "nodular ground glass opacity": "结节状磨玻璃影",
        "nodular ground glass opacities": "结节状磨玻璃影",
        "nodular ground glass densities": "结节状磨玻璃密度影",
        "patchy ground glass opacities": "斑片状磨玻璃影",
        "subpleural ground glass opacity": "胸膜下磨玻璃影",
        "subpleural ground glass opacities": "胸膜下磨玻璃影",
        "linear atelectasis": "线状肺不张",
        "subsegmental atelectasis": "亚段性肺不张",
        "atelectatic changes": "肺不张样改变",
        "atelectasis": "肺不张",
        "emphysematous changes": "肺气肿样改变",
        "minimal emphysematous changes": "轻微肺气肿样改变",
        "mild emphysematous changes": "轻度肺气肿样改变",
        "diffuse emphysematous changes": "弥漫性肺气肿样改变",
        "paraseptal emphysema": "间隔旁肺气肿",
        "centriacinar emphysema": "腺泡中央型肺气肿",
        "emphysema": "肺气肿",
        "bronchiectasis": "支气管扩张",
        "traction bronchiectasis": "牵拉性支气管扩张",
        "cystic bronchiectasis": "囊性支气管扩张",
        "peribronchial thickening": "支气管周围增厚",
        "bronchial wall thickening": "支气管壁增厚",
        "interlobular septal thickening": "小叶间隔增厚",
        "intralobular septal thickening": "小叶内间隔增厚",
        "septal thickening": "间隔增厚",
        "pleural effusion": "胸腔积液",
        "minimal pleural effusion": "少量胸腔积液",
        "pneumothorax": "气胸",
        "air bronchograms": "空气支气管征",
        "air bronchogram": "空气支气管征",
        "air cyst": "含气囊腔",
        "bleb formation": "肺小疱形成",
        "blebs": "肺小疱",
        "bleb": "肺小疱",
        "bullae": "肺大疱",
        "bulla": "肺大疱",
        "bullous": "肺大疱性",
        "tree in bud appearance": "树芽征表现",
        "tree-in-bud appearance": "树芽征表现",
        "tree in bud pattern": "树芽征模式",
        "tree-in-bud pattern": "树芽征模式",
        "tree in bud opacities": "树芽征样影",
        "tree-in-bud opacities": "树芽征样影",
        "mosaic attenuation pattern": "马赛克衰减模式",
        "mosaic attenuation": "马赛克衰减",
        "crazy paving pattern": "铺路石征",
        "halo sign": "晕征",
        "ground-glass halo": "磨玻璃晕",
        "ground glass halo": "磨玻璃晕",
        "consolidative density": "实变性密度影",
        "consolidative densities": "实变性密度影",
        "consolidation area": "实变区",
        "consolidation areas": "实变区",
        "areas of consolidation": "实变区",
        "nodular consolidation": "结节状实变",
        "nodular consolidations": "结节状实变",
        "consolidations": "实变",
        "consolidation": "实变",
        "pneumonic infiltration": "肺炎性浸润",
        "pneumonic infiltrates": "肺炎性浸润",
        "infiltration areas": "浸润区",
        "infiltrates": "浸润影",
        "infiltration": "浸润",
        "fibroatelectatic changes": "纤维肺不张样改变",
        "fibrotic scarring": "纤维性瘢痕",
        "fibrotic changes": "纤维化改变",
        "fibrosis": "纤维化",
        "scarring": "瘢痕",
        "pleuroparenchymal scarring": "胸膜肺实质瘢痕",
        "pleuroparenchymal density increases": "胸膜肺实质密度增高",
        "reticulonodular scarring": "网状结节样瘢痕",
        "reticulonodular infiltrates": "网状结节样浸润",
        "reticulonodular density increases": "网状结节样密度增高",
        "honeycomb appearance": "蜂窝样表现",
        "mass lesion": "肿块性病灶",
        "lesion": "病灶",
        "nodular lesions": "结节性病灶",
        "nonspecific parenchymal nodule": "非特异性肺实质结节",
        "nonspecific parenchymal nodules": "非特异性肺实质结节",
        "nonspecific pulmonary nodules": "非特异性肺结节",
        "pulmonary nodules": "肺结节",
        "pulmonary nodule": "肺结节",
        "parenchymal nodule": "肺实质结节",
        "parenchymal nodules": "肺实质结节",
        "calcified nodule": "钙化结节",
        "calcified nodules": "钙化结节",
        "calcific nodule": "钙化结节",
        "calcific nodules": "钙化结节",
        "micronodular infiltrates": "微结节样浸润",
        "micronodules": "微结节",
        "nodule": "结节",
        "nodules": "结节",
        "opacities": "阴影",
        "opacity": "阴影",
        "densities": "密度影",
        "density": "密度影",
        "increased density": "密度增高",
        "density increases": "密度增高",
        "parenchymal findings": "肺实质所见",
        "lung parenchyma": "肺实质",
        "subpleural area": "胸膜下区域",
        "subpleural region": "胸膜下区域",
        "areas": "区域",
        "area": "区域",
        "regions": "区域",
        "region": "区域",
        "portions": "部分",
        "portion": "部分",
        "parts": "部分",
        "part": "部分",
        "levels": "水平",
        "level": "水平",
        "zones": "带",
        "zone": "带",
        "subpleural": "胸膜下",
        "peripheral subpleural": "外周胸膜下",
        "peripheral": "外周",
        "central": "中央",
        "paramediastinal": "纵隔旁",
        "paracardiac": "心旁",
        "peribronchovascular": "支气管血管束周围",
        "peribronchial": "支气管周围",
        "intraparenchymal": "肺实质内",
        "bronchovascular structure": "支气管血管结构",
        "bronchovascular structures": "支气管血管结构",
        "pleural retraction": "胸膜牵拉",
        "volume loss": "容积减小",
        "structural distortion": "结构扭曲",
        "surrounding parenchyma": "周围肺实质",
        "surrounding ground glass": "周围磨玻璃影",
        "surrounding": "周围",
        "adjacent to": "邻近",
        "along": "沿",
        "within": "位于",
        "both lungs": "双肺",
        "bilateral lungs": "双肺",
        "bilaterally": "双侧",
        "bilateral": "双侧",
        "on the right": "右侧",
        "on the left": "左侧",
        "right side": "右侧",
        "left side": "左侧",
        "right lung": "右肺",
        "left lung": "左肺",
        "both lower lobes": "双下叶",
        "both upper lobes": "双上叶",
        "upper lobes": "上叶",
        "lower lobes": "下叶",
        "middle lobes": "中叶",
        "both lung apices": "双肺尖",
        "lung apices": "肺尖",
        "apices": "肺尖",
        "apex": "肺尖",
        "pleural spaces": "胸膜腔",
        "pleural space": "胸膜腔",
        "pleura": "胸膜",
        "diaphragm": "膈肌",
        "fissure": "叶间裂",
        "major fissure": "主裂",
        "minor fissure": "副裂",
        "horizontal fissure": "水平裂",
        "hilum": "肺门",
        "hilar": "肺门",
        "mediastinum": "纵隔",
        "aortic arch": "主动脉弓",
        "chest wall": "胸壁",
        "thorax": "胸廓",
        "subcutaneous adipose tissue": "皮下脂肪组织",
        "right hemithorax": "右侧胸腔",
        "left hemithorax": "左侧胸腔",
        "irregularly circumscribed": "边界不规则",
        "irregularly bordered": "边界不规则",
        "irregular borders": "边界不规则",
        "irregular": "不规则",
        "well circumscribed": "边界清楚",
        "well-circumscribed": "边界清楚",
        "well defined": "界限清楚",
        "well-defined": "界限清楚",
        "indistinct borders": "边界不清",
        "indistinctly circumscribed": "边界不清",
        "smooth contours": "轮廓光滑",
        "spiculated projections": "毛刺样突起",
        "spiculated": "毛刺样",
        "calcified": "钙化",
        "calcifications": "钙化",
        "cavitary": "空洞性",
        "thin-walled cavitary": "薄壁空洞性",
        "solid appearance": "实性表现",
        "semisolid": "半实性",
        "subsolid": "亚实性",
        "linear": "线状",
        "patchy": "斑片状",
        "nodular": "结节状",
        "focal": "局灶性",
        "diffuse": "弥漫性",
        "widespread": "广泛",
        "prominent": "明显",
        "more prominently": "更明显",
        "more prominent": "更明显",
        "predominantly": "主要",
        "minimal": "轻微",
        "mild": "轻度",
        "moderate": "中度",
        "severe": "重度",
        "slight": "轻度",
        "significant": "显著",
        "extensive": "范围较广",
        "occasional": "少数",
        "several": "数个",
        "multiple": "多个",
        "a few": "少数",
        "few": "少数",
        "two": "两个",
        "solitary": "单发",
        "single": "单个",
        "cluster": "簇状",
        "subcentimeter": "亚厘米",
        "largest": "最大",
        "measuring approximately": "测量约",
        "measuring up to": "最大测量约",
        "measuring": "测量",
        "approximately": "约",
        "in diameter": "直径",
        "up to": "达",
        "less than": "小于",
        "smaller than": "小于",
        "maximal thickness": "最大厚度",
        "deepest part": "最深处",
        "newly developed": "新发",
        "newly developing": "新近出现",
        "newly revealed": "新发现",
        "newly emerged": "新出现",
        "newly observed": "新观察到",
        "newly appeared": "新出现",
        "new": "新发",
        "stable": "稳定",
        "unchanged": "无变化",
        "progression": "进展",
        "progressed": "进展",
        "regression": "消退",
        "regressed": "缩小/消退",
        "resolved": "已吸收/消退",
        "decrease": "减少",
        "decreased": "减少",
        "increase": "增加",
        "increasing": "增加",
        "previous examination": "既往检查",
        "previous examinations": "既往检查",
        "prior examination": "既往检查",
        "prior chest radiographs": "既往胸片",
        "compared to": "较",
        "previously": "此前",
        "previous": "既往",
        "nonspecific": "非特异性",
        "non-specific": "非特异性",
        "indeterminate": "性质不确定",
        "suspicious": "可疑",
        "probable": "可能",
        "possibly": "可能",
        "consistent with": "符合",
        "suggestive of": "提示",
        "atypical": "非典型",
    }
    for phrase, zh in direct_phrases.items():
        add_phrase(phrases, phrase, zh)

    ordered = sorted(phrases.items(), key=lambda item: len(item[0]), reverse=True)
    return [(phrase_pattern(phrase), replacement) for phrase, replacement in ordered]


FUNCTION_RULES = [
    (re.compile(r"\bas well as\b", re.IGNORECASE), "以及"),
    (re.compile(r"\bin the form of\b", re.IGNORECASE), "呈"),
    (re.compile(r"\bincluding\b", re.IGNORECASE), "包括"),
    (re.compile(r"\binvolving\b", re.IGNORECASE), "累及"),
    (re.compile(r"\bcharacterized by\b", re.IGNORECASE), "表现为"),
    (re.compile(r"\bbased on\b", re.IGNORECASE), "基于"),
    (re.compile(r"\bcausing\b", re.IGNORECASE), "导致"),
    (re.compile(r"\bcontaining\b", re.IGNORECASE), "含有"),
    (re.compile(r"\bextending to\b", re.IGNORECASE), "延伸至"),
    (re.compile(r"\bextending into\b", re.IGNORECASE), "延伸入"),
    (re.compile(r"\bmore pronounced in\b", re.IGNORECASE), "在...更明显于"),
    (re.compile(r"\bparticularly in\b", re.IGNORECASE), "尤其位于"),
    (re.compile(r"\bespecially in\b", re.IGNORECASE), "尤其位于"),
    (re.compile(r"\bat the junction of\b", re.IGNORECASE), "位于交界处"),
    (re.compile(r"\bat the level of\b", re.IGNORECASE), "位于...水平"),
    (re.compile(r"\bat the\b", re.IGNORECASE), "于"),
    (re.compile(r"\bin both\b", re.IGNORECASE), "位于双侧"),
    (re.compile(r"\bin the\b", re.IGNORECASE), "位于"),
    (re.compile(r"\bin\b", re.IGNORECASE), "位于"),
    (re.compile(r"\bwith\b", re.IGNORECASE), "伴有"),
    (re.compile(r"\bwithout\b", re.IGNORECASE), "不伴"),
    (re.compile(r"\bof the\b", re.IGNORECASE), "的"),
    (re.compile(r"\bof\b", re.IGNORECASE), "的"),
    (re.compile(r"\band\b", re.IGNORECASE), "和"),
    (re.compile(r"\bor\b", re.IGNORECASE), "或"),
    (re.compile(r"\bto\b", re.IGNORECASE), "至"),
    (re.compile(r"\bfrom\b", re.IGNORECASE), "自"),
    (re.compile(r"\bon\b", re.IGNORECASE), "在"),
    (re.compile(r"\ba\b", re.IGNORECASE), ""),
    (re.compile(r"\ban\b", re.IGNORECASE), ""),
    (re.compile(r"\bthe\b", re.IGNORECASE), ""),
]


EXACT_TRANSLATIONS = {
    "subcentimeter nonspecific nodules in both lungs": "双肺亚厘米非特异性结节（subcentimeter nonspecific nodules）",
    "emphysematous changes in both lungs": "双肺肺气肿样改变（emphysematous changes）",
    "ground-glass opacities in both lungs": "双肺磨玻璃影（ground-glass opacities）",
    "ground glass opacities in both lungs": "双肺磨玻璃影（ground glass opacities）",
    "linear atelectasis in the medial segment of the right middle lobe": "右中叶内侧段线状肺不张（linear atelectasis）",
    "subpleural ground glass opacity in the posterobasal segment of the right lower lobe": "右下叶后基底段胸膜下磨玻璃影（subpleural ground glass opacity）",
}


def clean_translation(text: str) -> str:
    text = text.replace(",", "，").replace(";", "；")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([，；])\s*", r"\1", text)
    text = re.sub(r"\s+([）])", r"\1", text)
    text = re.sub(r"([（])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ，；")


def translate_prompt(prompt: str, translation_rules: list[tuple[re.Pattern, str]]) -> str:
    key = prompt.lower().strip()
    if key in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[key]
    text = key
    text = text.replace("ground-glass", "ground glass")
    text = text.replace("tree-in-bud", "tree in bud")
    text = text.replace("non-specific", "nonspecific")
    text = text.replace("well-defined", "well defined")
    text = text.replace("well-circumscribed", "well circumscribed")
    text = apply_outside_parentheses(text, translation_rules)
    text = apply_outside_parentheses(text, FUNCTION_RULES)
    text = clean_translation(text)
    return text


@dataclass(frozen=True)
class TermSpec:
    concept: str
    zh: str
    category: str
    patterns: tuple[str, ...]


def term(concept: str, zh: str, category: str, *patterns: str) -> TermSpec:
    return TermSpec(concept, zh, category, patterns)


TERM_SPECS = [
    term("Nodule", "结节", "Disease / Finding", r"\bnodules?\b", r"\bmicronodules?\b"),
    term("Pulmonary nodule", "肺结节", "Disease / Finding", r"\bpulmonary nodules?\b"),
    term("Ground-glass opacity", "磨玻璃影", "Disease / Finding", r"\bground[- ]glass opacit(?:y|ies)\b", r"\bground[- ]glass densit(?:y|ies)\b", r"\bground[- ]glass appearance\b"),
    term("Consolidation", "实变", "Disease / Finding", r"\bconsolidations?\b", r"\bconsolidative densit(?:y|ies)\b"),
    term("Atelectasis", "肺不张", "Disease / Finding", r"\batelectasis\b", r"\batelectatic changes\b"),
    term("Emphysema", "肺气肿", "Disease / Finding", r"\bemphysema\b", r"\bemphysematous changes\b"),
    term("Bronchiectasis", "支气管扩张", "Disease / Finding", r"\bbronchiectasis\b"),
    term("Scarring", "瘢痕", "Disease / Finding", r"\bscarring\b", r"\bscar\b"),
    term("Fibrosis", "纤维化", "Disease / Finding", r"\bfibrosis\b", r"\bfibrotic\b"),
    term("Infiltration", "浸润", "Disease / Finding", r"\binfiltrat(?:e|es|ion|ions)\b"),
    term("Pleural effusion", "胸腔积液", "Disease / Finding", r"\bpleural effusion\b", r"\beffusion\b"),
    term("Pneumothorax", "气胸", "Disease / Finding", r"\bpneumothorax\b"),
    term("Air cyst / bleb / bulla", "含气囊腔/肺小疱/肺大疱", "Disease / Finding", r"\bair cyst\b", r"\bblebs?\b", r"\bbulla\w*\b", r"\bbullous\b"),
    term("Cavitary lesion", "空洞性病灶", "Disease / Finding", r"\bcavitary\b", r"\bcavitation\b", r"\bcavity\b"),
    term("Mass / lesion", "肿块/病灶", "Disease / Finding", r"\bmass lesion\b", r"\blesions?\b"),
    term("Thickening", "增厚", "Disease / Finding", r"\bthickening\b", r"\bthickenings\b"),
    term("Air bronchogram", "空气支气管征", "Disease / Finding", r"\bair bronchograms?\b"),
    term("Tree-in-bud", "树芽征", "Disease / Finding", r"\btree[- ]in[- ]bud\b", r"\bbud branch\b"),
    term("Mosaic attenuation", "马赛克衰减", "Disease / Finding", r"\bmosaic attenuation\b"),
    term("Crazy paving", "铺路石征", "Disease / Finding", r"\bcrazy paving\b"),
    term("Halo sign", "晕征", "Disease / Finding", r"\bhalo sign\b", r"\bground[- ]glass halo\b"),
    term("Lung", "肺", "Anatomy / Location", r"\blungs?\b", r"\bpulmonary\b"),
    term("Lung parenchyma", "肺实质", "Anatomy / Location", r"\blung parenchyma\b", r"\bparenchymal\b"),
    term("Upper lobe", "上叶", "Anatomy / Location", r"\bupper lobes?\b"),
    term("Middle lobe", "中叶", "Anatomy / Location", r"\bmiddle lobe\b"),
    term("Lower lobe", "下叶", "Anatomy / Location", r"\blower lobes?\b"),
    term("Lingula / lingular segment", "舌叶/舌段", "Anatomy / Location", r"\blingula\b", r"\blingular segment\b"),
    term("Pleura / pleural space", "胸膜/胸膜腔", "Anatomy / Location", r"\bpleura\b", r"\bpleural\b"),
    term("Diaphragm", "膈肌", "Anatomy / Location", r"\bdiaphragm\b", r"\bsubdiaphragmatic\b"),
    term("Fissure", "叶间裂", "Anatomy / Location", r"\bfissure\b", r"\bfissures\b"),
    term("Hilum / hilar", "肺门", "Anatomy / Location", r"\bhilum\b", r"\bhilar\b"),
    term("Mediastinum", "纵隔", "Anatomy / Location", r"\bmediastin\w*\b"),
    term("Bronchus / bronchial", "支气管", "Anatomy / Location", r"\bbronch\w*\b"),
    term("Chest wall / thorax", "胸壁/胸廓", "Anatomy / Location", r"\bchest wall\b", r"\bthorax\b", r"\bhemithorax\b"),
    term("Apical / apex", "尖段/肺尖", "Lung Segment", r"\bapical\b", r"\bapices\b", r"\bapex\b"),
    term("Apicoposterior", "尖后段", "Lung Segment", r"\bapicoposterior\b"),
    term("Anterior", "前段", "Lung Segment", r"\banterior\b"),
    term("Posterior", "后段", "Lung Segment", r"\bposterior\b"),
    term("Superior", "上段", "Lung Segment", r"\bsuperior\b"),
    term("Inferior", "下段", "Lung Segment", r"\binferior\b"),
    term("Basal", "基底段", "Lung Segment", r"\bbasal\b"),
    term("Anterobasal", "前基底段", "Lung Segment", r"\banterobasal\b", r"\banterior basal\b"),
    term("Posterobasal", "后基底段", "Lung Segment", r"\bposterobasal\b", r"\bposterior basal\b"),
    term("Laterobasal", "外侧基底段", "Lung Segment", r"\blaterobasal\b", r"\blateral basal\b"),
    term("Mediobasal", "内侧基底段", "Lung Segment", r"\bmediobasal\b", r"\bmedial basal\b"),
    term("Medial", "内侧", "Lung Segment", r"\bmedial\b"),
    term("Lateral", "外侧", "Lung Segment", r"\blateral\b"),
    term("Right", "右侧", "Laterality", r"\bright\b"),
    term("Left", "左侧", "Laterality", r"\bleft\b"),
    term("Both lungs", "双肺", "Laterality", r"\bboth lungs\b"),
    term("Bilateral", "双侧", "Laterality", r"\bbilateral\b", r"\bbilaterally\b"),
    term("Subpleural", "胸膜下", "Spatial Relation", r"\bsubpleural\b"),
    term("Peribronchial", "支气管周围", "Spatial Relation", r"\bperibronchial\b"),
    term("Peribronchovascular", "支气管血管束周围", "Spatial Relation", r"\bperibronchovascular\b"),
    term("Paramediastinal", "纵隔旁", "Spatial Relation", r"\bparamediastinal\b"),
    term("Paracardiac", "心旁", "Spatial Relation", r"\bparacardiac\b"),
    term("Peripheral", "外周", "Spatial Relation", r"\bperipheral\b", r"\bperipherally\b"),
    term("Central", "中央", "Spatial Relation", r"\bcentral\b", r"\bcentrally\b"),
    term("Adjacent", "邻近", "Spatial Relation", r"\badjacent\b"),
    term("Surrounding", "周围", "Spatial Relation", r"\bsurrounding\b"),
    term("Within", "内部/位于", "Spatial Relation", r"\bwithin\b"),
    term("Around", "周围", "Spatial Relation", r"\baround\b"),
    term("Nodular", "结节状", "Morphology / Imaging Characteristic", r"\bnodular\b"),
    term("Linear", "线状", "Morphology / Imaging Characteristic", r"\blinear\b"),
    term("Patchy", "斑片状", "Morphology / Imaging Characteristic", r"\bpatchy\b"),
    term("Irregular", "不规则", "Morphology / Imaging Characteristic", r"\birregular\w*\b"),
    term("Spiculated", "毛刺样", "Morphology / Imaging Characteristic", r"\bspiculated\b"),
    term("Calcified / calcific", "钙化", "Morphology / Imaging Characteristic", r"\bcalcif\w*\b"),
    term("Cavitary", "空洞性", "Morphology / Imaging Characteristic", r"\bcavitary\b", r"\bcavitation\b"),
    term("Reticulonodular", "网状结节样", "Morphology / Imaging Characteristic", r"\breticulonodular\b"),
    term("Mosaic attenuation pattern", "马赛克衰减模式", "Morphology / Imaging Characteristic", r"\bmosaic attenuation pattern\b"),
    term("Indistinct border", "边界不清", "Morphology / Imaging Characteristic", r"\bindistinct\b", r"\bill-defined\b"),
    term("Well-defined / circumscribed", "界限清楚/边界清楚", "Morphology / Imaging Characteristic", r"\bwell[- ]defined\b", r"\bwell[- ]circumscribed\b"),
    term("Smooth contour", "轮廓光滑", "Morphology / Imaging Characteristic", r"\bsmooth contours?\b"),
    term("Structural distortion", "结构扭曲", "Morphology / Imaging Characteristic", r"\bstructural distortion\b"),
    term("Volume loss", "容积减小", "Morphology / Imaging Characteristic", r"\bvolume loss\b"),
    term("Pleural retraction", "胸膜牵拉", "Morphology / Imaging Characteristic", r"\bpleural retraction\b"),
    term("Subcentimeter", "亚厘米", "Size / Quantity", r"\bsubcentimeter\b"),
    term("Millimeter measurement", "毫米测量", "Size / Quantity", r"\b\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?mm\b", r"\bmm\b"),
    term("Centimeter measurement", "厘米测量", "Size / Quantity", r"\b\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?cm\b", r"\bcm\b"),
    term("Measuring / diameter", "测量/直径", "Size / Quantity", r"\bmeasuring\b", r"\bdiameter\b"),
    term("Largest", "最大", "Size / Quantity", r"\blargest\b"),
    term("Less than / smaller than", "小于", "Size / Quantity", r"\bless than\b", r"\bsmaller than\b"),
    term("Multiple", "多个", "Size / Quantity", r"\bmultiple\b"),
    term("Several", "数个", "Size / Quantity", r"\bseveral\b"),
    term("A few / few", "少数", "Size / Quantity", r"\ba few\b", r"\bfew\b"),
    term("Solitary / single", "单发/单个", "Size / Quantity", r"\bsolitary\b", r"\bsingle\b"),
    term("Two", "两个", "Size / Quantity", r"\btwo\b"),
    term("Cluster", "簇状", "Size / Quantity", r"\bcluster\b"),
    term("Minimal", "轻微", "Severity / Extent", r"\bminimal\b"),
    term("Mild", "轻度", "Severity / Extent", r"\bmild\b"),
    term("Moderate", "中度", "Severity / Extent", r"\bmoderate\b"),
    term("Severe", "重度", "Severity / Extent", r"\bsevere\b"),
    term("Diffuse", "弥漫性", "Severity / Extent", r"\bdiffuse\b"),
    term("Widespread", "广泛", "Severity / Extent", r"\bwidespread\b"),
    term("Focal", "局灶性", "Severity / Extent", r"\bfocal\b"),
    term("Prominent", "明显", "Severity / Extent", r"\bprominent\b", r"\bprominently\b"),
    term("Slight", "轻度/轻微", "Severity / Extent", r"\bslight\b", r"\bslightly\b"),
    term("Significant", "显著", "Severity / Extent", r"\bsignificant\b", r"\bsignificantly\b"),
    term("Extensive", "范围较广", "Severity / Extent", r"\bextensive\b"),
    term("Predominant", "为主", "Severity / Extent", r"\bpredominant\w*\b"),
    term("Occasional", "少数/偶发", "Severity / Extent", r"\boccasional\b"),
    term("New / newly developed", "新发/新近出现", "Temporal / Comparison", r"\bnew\b", r"\bnewly\b"),
    term("Stable", "稳定", "Temporal / Comparison", r"\bstable\b"),
    term("Progression", "进展", "Temporal / Comparison", r"\bprogress(?:ion|ed)?\b"),
    term("Regression", "消退/缩小", "Temporal / Comparison", r"\bregress(?:ion|ed)?\b"),
    term("Previous / prior", "既往", "Temporal / Comparison", r"\bprevious(?:ly)?\b", r"\bprior\b"),
    term("Compared to", "较/相比", "Temporal / Comparison", r"\bcompared to\b"),
    term("Unchanged", "无变化", "Temporal / Comparison", r"\bunchanged\b"),
    term("Resolved / resorption", "吸收/消退", "Temporal / Comparison", r"\bresolved\b", r"\bresorption\b"),
    term("Decrease", "减少", "Temporal / Comparison", r"\bdecrease(?:d)?\b"),
    term("Increase", "增加/增高", "Temporal / Comparison", r"\bincrease(?:d|s|ing)?\b"),
    term("Nonspecific", "非特异性", "Uncertainty / Diagnostic Qualifier", r"\bnon[- ]?specific\b", r"\bnonspecific\b"),
    term("Indeterminate", "性质不确定", "Uncertainty / Diagnostic Qualifier", r"\bindeterminate\b"),
    term("Suspicious", "可疑", "Uncertainty / Diagnostic Qualifier", r"\bsuspicious\b"),
    term("Probable / possible", "可能", "Uncertainty / Diagnostic Qualifier", r"\bprobable\b", r"\bpossibly\b", r"\bpossible\b"),
    term("Consistent with", "符合", "Uncertainty / Diagnostic Qualifier", r"\bconsistent with\b"),
    term("Suggestive of", "提示", "Uncertainty / Diagnostic Qualifier", r"\bsuggestive of\b"),
    term("Atypical", "非典型", "Uncertainty / Diagnostic Qualifier", r"\batypical\b"),
]

CATEGORY_ORDER = [
    "Disease / Finding",
    "Anatomy / Location",
    "Lung Segment",
    "Laterality",
    "Spatial Relation",
    "Morphology / Imaging Characteristic",
    "Size / Quantity",
    "Severity / Extent",
    "Temporal / Comparison",
    "Uncertainty / Diagnostic Qualifier",
]


def build_counts(instances: list[dict]) -> tuple[dict[str, Counter], Counter]:
    split_counts = {split: Counter() for split in SPLITS}
    total_counts = Counter()
    for instance in instances:
        prompt = instance["prompt"]
        split_counts[instance["split"]][prompt] += 1
        total_counts[prompt] += 1
    return split_counts, total_counts


def build_vocabulary_rows(split_counts: dict[str, Counter]) -> list[list[object]]:
    rows = []
    category_rank = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    compiled_specs = [
        (
            spec,
            [re.compile(pattern, re.IGNORECASE) for pattern in spec.patterns],
        )
        for spec in TERM_SPECS
    ]
    for spec, patterns in compiled_specs:
        counts = Counter()
        raw_variants = Counter()
        examples = []
        for split in SPLITS:
            for prompt, count in split_counts[split].items():
                matches = []
                for pattern in patterns:
                    matches.extend(match.group(0).lower() for match in pattern.finditer(prompt))
                if not matches:
                    continue
                counts[split] += count
                raw_variants.update({match: count for match in set(matches)})
                if len(examples) < 3 and prompt not in examples:
                    examples.append(prompt)
        total = sum(counts[split] for split in SPLITS)
        if total == 0:
            continue
        raw_text = "; ".join(f"{variant} ({count})" for variant, count in raw_variants.most_common(12))
        example_text = " / ".join(examples)
        rows.append(
            [
                spec.concept,
                spec.zh,
                spec.category,
                raw_text,
                counts["train"],
                counts["val"],
                counts["test"],
                total,
                example_text,
            ]
        )
    rows.sort(key=lambda row: (category_rank[row[2]], -int(row[7]), row[0].lower()))
    return rows


def write_readme(path: Path, metadata_path: Path, split_case_counts: dict[str, int], split_prompt_counts: dict[str, int], split_unique_counts: dict[str, int], total_unique: int) -> None:
    total_cases = sum(split_case_counts.values())
    total_prompt_instances = sum(split_prompt_counts.values())
    content = f"""---
created: {DOC_DATE}
updated: {DOC_DATE}
status: active
---

# ReXGroundingCT Prompt Illustration

This folder documents the English free-text prompts used in the ReXGroundingCT
train/val/test splits and provides Chinese translations for prompt
understanding.

Source metadata:

```text
{metadata_path}
```

These translations are for research reading, anatomy learning, and prompt
analysis. They are not certified clinical report translations.

## Split Counts

| Split | Cases | Prompt instances | Unique prompts |
| --- | ---: | ---: | ---: |
| train | {split_case_counts['train']} | {split_prompt_counts['train']} | {split_unique_counts['train']} |
| val | {split_case_counts['val']} | {split_prompt_counts['val']} | {split_unique_counts['val']} |
| test | {split_case_counts['test']} | {split_prompt_counts['test']} | {split_unique_counts['test']} |
| total | {total_cases} | {total_prompt_instances} | {total_unique} |

## Folder Map

- `{PROMPT_UNIQUE_DOC}`
  - One row per unique English prompt.
  - Includes Chinese translation and train/val/test/total appearance counts.

- `{PROMPT_INSTANCES_DOC}`
  - One row per prompt occurrence.
  - Preserves split, case name, finding ID, category, and entity count when
    available.

- `{PROMPT_VOCAB_DOC}`
  - Structured glossary of disease, anatomy, location, morphology, size,
    severity, temporal, and uncertainty terms.
  - Includes raw surface variants, split counts, and example prompts.

## Translation Style

Chinese translations use radiology-oriented phrasing and keep important English
terms in parentheses. The translation pass is deterministic and glossary-based,
so uncommon long prompts may remain partially bilingual when an English fragment
does not map cleanly to the glossary.
"""
    path.write_text(content, encoding="utf-8")


def validate_docs(
    instances: list[dict],
    split_counts: dict[str, Counter],
    total_counts: Counter,
    translations: dict[str, str],
    vocab_rows: list[list[object]],
) -> None:
    for split, expected in EXPECTED_PROMPT_COUNTS.items():
        actual = sum(split_counts[split].values())
        if actual != expected:
            raise AssertionError(f"{split}: expected {expected} prompt instances, got {actual}")
    total_prompt_instances = len(instances)
    if total_prompt_instances != sum(EXPECTED_PROMPT_COUNTS.values()):
        raise AssertionError(f"expected {sum(EXPECTED_PROMPT_COUNTS.values())} total instances, got {total_prompt_instances}")
    if len(total_counts) != EXPECTED_UNIQUE_PROMPTS:
        raise AssertionError(f"expected {EXPECTED_UNIQUE_PROMPTS} unique prompts, got {len(total_counts)}")
    missing = [prompt for prompt in total_counts if not translations.get(prompt)]
    if missing:
        raise AssertionError(f"missing translations for {len(missing)} prompts")
    for instance in instances:
        if instance["prompt"] not in translations:
            raise AssertionError(f"instance prompt missing translation: {instance['prompt']}")
    split_totals = {split: sum(split_counts[split].values()) for split in SPLITS}
    for row in vocab_rows:
        for split_index, split in enumerate(SPLITS, start=4):
            if int(row[split_index]) > split_totals[split]:
                raise AssertionError(f"vocabulary count exceeds split total for {row[0]} / {split}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    instances = iter_instances(metadata)
    split_counts, total_counts = build_counts(instances)
    translation_rules = build_translation_rules()
    translations = {prompt: translate_prompt(prompt, translation_rules) for prompt in total_counts}

    sorted_prompts = sorted(total_counts, key=lambda prompt: (-total_counts[prompt], prompt.lower()))
    unique_rows = []
    for index, prompt in enumerate(sorted_prompts, start=1):
        unique_rows.append(
            [
                f"P{index:05d}",
                prompt,
                translations[prompt],
                split_counts["train"][prompt],
                split_counts["val"][prompt],
                split_counts["test"][prompt],
                total_counts[prompt],
            ]
        )

    instance_rows = [
        [
            instance["split"],
            instance["case_name"],
            instance["finding_id"],
            instance["category"],
            instance["entity_count"],
            instance["prompt"],
            translations[instance["prompt"]],
        ]
        for instance in instances
    ]

    vocab_rows = build_vocabulary_rows(split_counts)
    validate_docs(instances, split_counts, total_counts, translations, vocab_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_case_counts = {split: len(metadata[split]) for split in SPLITS}
    split_prompt_counts = {split: sum(split_counts[split].values()) for split in SPLITS}
    split_unique_counts = {split: len(split_counts[split]) for split in SPLITS}

    write_readme(
        args.output_dir / "README.md",
        args.metadata,
        split_case_counts,
        split_prompt_counts,
        split_unique_counts,
        len(total_counts),
    )
    write_table(
        args.output_dir / PROMPT_UNIQUE_DOC,
        ["ID", "English prompt", "中文翻译", "train", "val", "test", "total"],
        unique_rows,
    )
    write_table(
        args.output_dir / PROMPT_INSTANCES_DOC,
        ["split", "case_name", "finding_id", "category", "entity_count", "English prompt", "中文翻译"],
        instance_rows,
    )
    write_table(
        args.output_dir / PROMPT_VOCAB_DOC,
        ["English concept", "中文", "Category", "Raw variants", "train", "val", "test", "total", "Example prompts"],
        vocab_rows,
    )

    print(f"Wrote docs to {args.output_dir}")
    print(f"Prompt instances: {len(instances)}")
    print(f"Unique prompts: {len(total_counts)}")
    print(f"Vocabulary rows: {len(vocab_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
