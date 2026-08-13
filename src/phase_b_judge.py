from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (OPENAI_API_KEY, OPENAI_BASE_URL, JUDGE_MODEL,
                    HUMAN_LABELS_PATH, ANSWERS_PATH, TEST_SET_PATH)


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


def _make_openai_client():
    from openai import OpenAI
    kw = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kw["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kw)


def quality_eval(question: str, answer: str, context: str = "", reference: str = "") -> int:
    """Evaluate a single answer as correct (1) or incorrect (0) for Cohen κ calculation.

    `context`   — các chunk chính sách được retrieve cho câu hỏi.
    `reference` — ground truth của test set (reference-based judging).

    Judge không có context/reference phải đoán các ngưỡng nội bộ (mua sắm >50 triệu
    cần CEO, phép năm v2024 = 15 ngày...) nên hay đánh sai câu trả lời đúng →
    agreement với human labels tụt mạnh. Xem so sánh κ trong reports/judge_results.json.
    """
    context_block = (
        f"Trích dẫn chính sách công ty được retrieve:\n{context}\n\n" if context else ""
    )
    reference_block = (
        f"Đáp án tham chiếu do chuyên gia HR soạn:\n{reference}\n\n" if reference else ""
    )
    prompt = (
        f"{context_block}{reference_block}"
        f"Câu hỏi: {question}\n\n"
        f"Câu trả lời cần đánh giá: {answer}\n\n"
        "Đánh giá câu trả lời là ĐÚNG (1) hay SAI (0):\n"
        "- ĐÚNG: khớp với đáp án tham chiếu / chính sách về các con số và kết luận "
        "chính, và trả lời đủ các phần mà câu hỏi yêu cầu. Ngắn gọn không phải là lỗi.\n"
        "- SAI: mâu thuẫn với chính sách, sai con số hoặc ngưỡng phê duyệt, trích "
        "phiên bản chính sách đã hết hiệu lực, hoặc bỏ sót hẳn một phần câu hỏi hỏi rõ.\n"
        "Không đánh SAI chỉ vì thiếu chi tiết phụ hoặc vì tài liệu không đủ để kiểm chứng.\n"
        'Trả lời JSON: {{"correct": true, "reason": "..."}}'
    )
    try:
        client = _make_openai_client()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá HR policy answers. Chỉ trả lời JSON."},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,   # judge phải deterministic thì κ mới tái lập được
            max_tokens=200,
        )
        data = json.loads(resp.choices[0].message.content)
        return 1 if data.get("correct", False) else 0
    except Exception:
        return 0


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí."""
    PROMPT_TEMPLATE = (
        "Bạn là một expert đánh giá chất lượng câu trả lời RAG.\n\n"
        "Câu hỏi: {question}\n\n"
        "Answer A:\n{answer_a}\n\n"
        "Answer B:\n{answer_b}\n\n"
        "Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.\n"
        'Trả lời JSON (chỉ JSON, không text khác):\n'
        '{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", '
        '"scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}'
    )
    try:
        client = _make_openai_client()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user",   "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0,   # loại bỏ nhiễu sampling, chỉ còn position bias thực
        )
        result = json.loads(resp.choices[0].message.content)
        # Normalize winner to uppercase
        result["winner"] = result.get("winner", "tie").upper()
        if result["winner"] not in {"A", "B", "TIE"}:
            result["winner"] = "tie"
        result["winner"] = result["winner"] if result["winner"] != "TIE" else "tie"
        result.setdefault("reasoning", "")
        result.setdefault("scores", {"A": 0.5, "B": 0.5})
        return result
    except Exception as e:
        print(f"⚠️  Judge API error: {e}")
        return {"winner": "tie", "reasoning": "", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán."""
    pass1     = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    swap_map     = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw["winner"], "tie")

    position_consistent = (pass1["winner"] == winner_pass2)
    final = pass1["winner"] if position_consistent else "tie"

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {"A": 0.5, "B": 0.5}),
        scores_pass2={
            "A": pass2_raw.get("scores", {}).get("B", 0.5),
            "B": pass2_raw.get("scores", {}).get("A", 0.5),
        },
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    n   = len(judge_labels)
    if n == 0:
        return 0.0
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    p_e = (
        judge_labels.count(1) / n * human_labels.count(1) / n
        + judge_labels.count(0) / n * human_labels.count(0) / n
    )
    if p_e == 1:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0,
            "position_bias_count": 0, "verbosity_details": {}, "interpretation": "",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged":        total,
        "position_bias_rate":  round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias":      round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive":  decisive,
        },
        "interpretation": interpretation,
    }


# ─── Verbosity probe (length-controlled) ─────────────────────────────────────

VERBOSITY_PROBE = [
    {
        "question": "Nhân viên chính thức được nghỉ bao nhiêu ngày phép năm?",
        "short_correct": "15 ngày phép năm (chính sách v2024).",
        "long_wrong": (
            "Theo quy định hiện hành của công ty về chế độ nghỉ phép, được xây dựng "
            "trên cơ sở Bộ luật Lao động và các văn bản nội bộ liên quan, mỗi nhân viên "
            "chính thức sẽ được hưởng tổng cộng 12 ngày phép năm có hưởng lương, số ngày "
            "này được cộng dồn theo tháng làm việc và có thể chuyển sang quý I năm sau "
            "nếu chưa sử dụng hết, đồng thời được ghi nhận đầy đủ trên hệ thống HRM."
        ),
    },
    {
        "question": "Mua thiết bị 55 triệu đồng cần ai phê duyệt?",
        "short_correct": "CEO phê duyệt, vì vượt ngưỡng 50 triệu.",
        "long_wrong": (
            "Quy trình mua sắm nội bộ được thiết kế theo nhiều cấp phê duyệt nhằm đảm bảo "
            "tính minh bạch và kiểm soát chi phí. Với một khoản chi có giá trị 55 triệu "
            "đồng, hồ sơ sẽ được lập bởi bộ phận đề xuất, chuyển qua bộ phận mua sắm rà "
            "soát báo giá, và sau đó Giám đốc phòng ban là cấp có thẩm quyền phê duyệt "
            "cuối cùng trước khi phát hành đơn đặt hàng cho nhà cung cấp."
        ),
    },
    {
        "question": "Nhân viên thử việc có được nghỉ phép năm không?",
        "short_correct": "Không. Thử việc chưa được nghỉ phép năm, nếu cần thì xin nghỉ không lương.",
        "long_wrong": (
            "Chính sách nhân sự của công ty luôn hướng tới việc đảm bảo quyền lợi cho mọi "
            "nhân viên ngay từ ngày đầu tiên gia nhập. Vì vậy, nhân viên trong giai đoạn "
            "thử việc vẫn được hưởng đầy đủ chế độ nghỉ phép năm theo tỷ lệ tương ứng với "
            "thời gian làm việc thực tế, và có thể đăng ký nghỉ trên hệ thống HRM giống "
            "như nhân viên chính thức, chỉ cần có sự đồng ý của quản lý trực tiếp."
        ),
    },
]


def verbosity_probe(probe: list[dict] = None) -> dict:
    """Đo verbosity bias theo cách có kiểm soát độ dài.

    `bias_report()` tính verbosity bias trên các cặp thật, nơi độ dài và chất lượng
    đi cùng nhau (câu trả lời tốt thường dài hơn) → số đo bị nhiễu. Probe này đảo
    ngược: câu NGẮN là câu ĐÚNG, câu DÀI là câu SAI. Judge chọn câu dài = bias thật.
    """
    probe = probe or VERBOSITY_PROBE
    cases, biased = [], 0
    for c in probe:
        # short_correct ở vị trí A, long_wrong ở vị trí B; swap-and-average khử position bias
        res = swap_and_average(c["question"], c["short_correct"], c["long_wrong"])
        picked_long = res.final_winner == "B"
        biased += int(picked_long)
        cases.append({
            "question":       c["question"],
            "short_correct":  c["short_correct"],
            "long_wrong":     c["long_wrong"],
            "len_ratio":      round(len(c["long_wrong"]) / len(c["short_correct"]), 1),
            "final_winner":   res.final_winner,
            "picked_long_wrong": picked_long,
            "position_consistent": res.position_consistent,
        })
    return {
        "n_cases":            len(cases),
        "picked_long_wrong":  biased,
        "verbosity_bias_rate": round(biased / len(cases), 3) if cases else 0.0,
        "note": "Câu dài luôn là câu SAI → tỷ lệ > 0 nghĩa là judge thực sự thiên vị độ dài.",
        "cases":              cases,
    }


# ─── Report ───────────────────────────────────────────────────────────────────

KAPPA_SCALE = [
    (0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"),
    (0.2, "fair"), (0.0, "slight"),
]


def kappa_interpretation(kappa: float) -> str:
    """Landis-Koch scale label cho giá trị κ."""
    for threshold, label in KAPPA_SCALE:
        if kappa >= threshold:
            return label
    return "poor (worse than chance)"


def _load_contexts_by_qid() -> dict[int, str]:
    """Map question_id → retrieved context (từ answers_50q.json, nếu đã chạy setup)."""
    if not os.path.exists(ANSWERS_PATH):
        return {}
    with open(ANSWERS_PATH, encoding="utf-8") as f:
        return {a["id"]: "\n\n".join(a.get("contexts", [])) for a in json.load(f)}


def _load_ground_truth_by_qid() -> dict[int, str]:
    """Map question_id → ground_truth (dùng làm answer B trong pairwise judge)."""
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        return {q["id"]: q["ground_truth"] for q in json.load(f)}


def save_phase_b_report(judge_results: list[JudgeResult], kappa: float,
                        judge_labels: list[int], human_labels: list[int],
                        bias: dict, per_question: list[dict],
                        path: str = "reports/judge_results.json",
                        kappa_ctx_only: float | None = None,
                        judge_labels_ctx_only: list[int] | None = None,
                        probe: dict | None = None) -> None:
    """Gom kết quả Phase B → reports/judge_results.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "judge_model": JUDGE_MODEL,
        "total_pairs": len(judge_results),
        "cohen_kappa": {
            "value":          round(kappa, 4),
            "interpretation": kappa_interpretation(kappa),
            "substantial":    kappa > 0.6,
            "n":              len(human_labels),
            "judge_labels":   judge_labels,
            "human_labels":   human_labels,
            "agreement_rate": round(
                sum(j == h for j, h in zip(judge_labels, human_labels)) / len(human_labels), 3
            ) if human_labels else 0.0,
        },
        "cohen_kappa_context_only": {
            "value":          round(kappa_ctx_only, 4) if kappa_ctx_only is not None else None,
            "interpretation": kappa_interpretation(kappa_ctx_only)
                              if kappa_ctx_only is not None else None,
            "judge_labels":   judge_labels_ctx_only,
            "note": "Judge chỉ thấy retrieved context, không thấy reference answer — "
                    "cho thấy grounding ảnh hưởng thế nào tới agreement với human.",
        },
        "bias_report": bias,
        "verbosity_probe": probe,
        "per_question": per_question,
        "pairwise_results": [
            {
                "question":            r.question,
                "answer_a":            r.answer_a,
                "answer_b":            r.answer_b,
                "winner_pass1":        r.winner_pass1,
                "winner_pass2":        r.winner_pass2,
                "final_winner":        r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning_pass1":     r.reasoning_pass1,
                "reasoning_pass2":     r.reasoning_pass2,
                "scores_pass1":        r.scores_pass1,
                "scores_pass2":        r.scores_pass2,
            }
            for r in judge_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"Human labels loaded: {len(human_labels)} questions")

    contexts_by_qid = _load_contexts_by_qid()
    gt_by_qid       = _load_ground_truth_by_qid()
    if not contexts_by_qid:
        print("⚠️  answers_50q.json chưa có — judge sẽ chạy không có retrieved context.")

    # --- Task 7: judge labels trên đúng 10 câu human đã gán nhãn ---
    # Chạy 2 cấu hình judge để đo ảnh hưởng của grounding lên agreement:
    #   (a) chỉ có retrieved context   (b) context + reference answer
    judge_labels: list[int] = []
    judge_labels_ctx_only: list[int] = []
    judge_results: list[JudgeResult] = []
    per_question: list[dict] = []

    for item in human_data:
        qid    = item["question_id"]
        q      = item["question"]
        ans    = item["model_answer"]
        ctx    = contexts_by_qid.get(qid, "")
        ref    = gt_by_qid.get(qid, "")

        label_ctx = quality_eval(q, ans, ctx)
        label     = quality_eval(q, ans, ctx, ref)
        judge_labels_ctx_only.append(label_ctx)
        judge_labels.append(label)

        # --- Tasks 5+6: pairwise model_answer (A) vs ground_truth (B), swap-and-average ---
        pair = swap_and_average(q, ans, ref or ans)
        judge_results.append(pair)

        per_question.append({
            "question_id":            qid,
            "question":               q,
            "model_answer":           ans,
            "judge_label":            label,
            "judge_label_ctx_only":   label_ctx,
            "human_label":            item["human_label"],
            "agree":                  label == item["human_label"],
            "human_note":             item["human_note"],
            "pairwise_winner":        pair.final_winner,
            "position_consistent":    pair.position_consistent,
        })
        print(f"  [q{qid:>2}] judge={label} (ctx-only={label_ctx}) human={item['human_label']} "
              f"{'✓' if label == item['human_label'] else '✗'} | "
              f"pairwise={pair.final_winner} (consistent={pair.position_consistent})")

    kappa          = cohen_kappa(judge_labels, human_labels)
    kappa_ctx_only = cohen_kappa(judge_labels_ctx_only, human_labels)
    print(f"\nCohen's κ (context + reference) = {kappa:.3f} → {kappa_interpretation(kappa)}"
          f"{' (bonus: κ > 0.6 ✓)' if kappa > 0.6 else ''}")
    print(f"Cohen's κ (context only)        = {kappa_ctx_only:.3f} → "
          f"{kappa_interpretation(kappa_ctx_only)}")

    bias = bias_report(judge_results)
    print(f"Position bias rate: {bias['position_bias_rate']:.1%} "
          f"({bias['position_bias_count']}/{bias['total_judged']})")
    print(f"Verbosity bias:     {bias['verbosity_bias']:.1%}")
    print(f"→ {bias['interpretation']}")

    print("\nVerbosity probe (câu ngắn = đúng, câu dài = sai)...")
    probe = verbosity_probe()
    print(f"  Judge chọn câu dài-nhưng-sai: {probe['picked_long_wrong']}/{probe['n_cases']} "
          f"→ verbosity bias thực = {probe['verbosity_bias_rate']:.0%}")

    save_phase_b_report(judge_results, kappa, judge_labels, human_labels, bias, per_question,
                        kappa_ctx_only=kappa_ctx_only,
                        judge_labels_ctx_only=judge_labels_ctx_only,
                        probe=probe)
