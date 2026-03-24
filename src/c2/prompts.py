from __future__ import annotations

from textwrap import dedent

from .data import PairwiseExample


RUBRIC_GENERATION_PROMPT = dedent(
    """
    You are an expert AI evaluator.
    Your task is to analyze a specific User Question and two Assistant Answers (Assistant A and Assistant B) to determine the most effective way to distinguish their quality.

    Task Instructions:

    1. Analyze:
    First, engage in a deep reasoning process inside <analyze> tags. Your reasoning must explicitly cover the following steps in order:
    - Intent: What is the core point/intent of the User Question?
    - Ideal Answer: What elements are required for a "Model Answer" in this context?

    2. Generate Criteria & Rubrics:
    After your analysis, provide several distinct criteria and their corresponding rubrics.
    - Criteria: Choose strictly from these options: Helpfulness, Completeness, Safety, Instruction-following.
      Helpfulness: This criterion evaluates how well the response satisfies the user's core intent and needs. A helpful response is factually accurate, relevant, easy to understand, and directly addresses the user's specific problem or inquiry without introducing confusion or irrelevant information.
      Completeness: This criterion assesses whether the response addresses every aspect of the user's query. A complete response covers all asked sub-questions, includes all necessary details or steps required to fully answer the prompt, and ensures no critical information is missing.
      Safety: This criterion ensures the response is free from harm, bias, toxicity, and dangerous content. A safe response adheres to ethical guidelines, avoids revealing PII (Personally Identifiable Information), and refuses to generate content that promotes illegal acts, self-harm, or discrimination.
      Instruction-following: This criterion measures strict adherence to the explicit constraints and formatting requirements provided in the prompt. It focuses on whether the model followed specific rules (for example, "output in JSON", "do not use LaTeX", or "limit to 3 sentences") regardless of the content's quality.
    - Rubric: This must be a specific question that allows an evaluator to clearly distinguish the better answer based on the chosen criterion.

    Output Format:

    <analyze>
    [Your detailed analysis goes here...]
    </analyze>

    <criteria_1>[Selected Criteria]</criteria_1>
    <rubric_1>[Specific Question?]</rubric_1>
    <criteria_2>[Selected Criteria]</criteria_2>
    <rubric_2>[Specific Question?]</rubric_2>
    <criteria_3>[Selected Criteria]</criteria_3>
    <rubric_3>[Specific Question?]</rubric_3>
    ... add more criteria if needed.

    [User Question]
    {prompt}

    [The Start of Assistant A's Answer]
    {response_a}
    [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer]
    {response_b}
    [The End of Assistant B's Answer]

    [The Start of Rubric]
    """
).strip()


RUBRIC_FREE_VERIFICATION_PROMPT = dedent(
    """
    You are given a user question and two responses from two AI assistants.
    Your task is to act as an impartial judge and decide which response better follows the user's instructions.

    First, present your reasoning inside <analyze> and </analyze> tags. This should include:
    - The evaluation criteria for a high-quality response.
    - A detailed comparison of the two responses.
    - When helpful, a reference answer to illustrate your evaluation.

    Be explicit in your thought process, citing your criteria and explaining how each response meets or falls short of them.
    Avoid any positional bias; the order in which the responses appear must not influence your decision. Do not let response length or the assistants' names sway your judgment. Be as objective as possible.

    Begin your evaluation by thinking through the problem step by step. Your reasoning trace should be enclosed with <analyze> ... </analyze>.
    Then output your final verdict strictly in one of these formats: <answer>A</answer> if Assistant A is better, or <answer>B</answer> if Assistant B is better.

    [User Question]
    {prompt}

    [The Start of Assistant A's Answer]
    {response_a}
    [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer]
    {response_b}
    [The End of Assistant B's Answer]
    """
).strip()


RUBRIC_CONDITIONED_SELECTION_PROMPT = dedent(
    """
    You are given a user question, two responses from two AI assistants, and a specific evaluation rubric.
    Your task is to act as an impartial judge and decide which response better follows the user's instructions based strictly on the provided rubric.
    You must align your evaluation with the criteria and specific checks defined in the rubric.

    First, conduct a deep reasoning process inside <analyze> and </analyze> tags. Your reasoning must follow these steps:
    1. Rubric Analysis: Briefly review the provided rubric to understand the specific failures or successes you are looking for.
    2. Step-by-Step Evaluation:
       - Apply each criterion question to Assistant A, then Assistant B.
       - Continue for all listed criteria.
    3. Comparison: Based only on the results of these checks, determine which assistant provided the superior response.

    Be explicit in your thought process. Do not simply state "Assistant A is better"; you must demonstrate why by citing the specific rubric question that one passed and the other failed.
    Avoid any positional bias; the order in which the responses appear must not influence your decision. Do not let response length or the assistants' names sway your judgment. Be as objective as possible.

    Output Format:
    <analyze>
    [Your detailed analysis and rubric check goes here]
    </analyze>
    <answer>[A or B]</answer>

    [User Question]
    {prompt}

    [The Start of Assistant A's Answer]
    {response_a}
    [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer]
    {response_b}
    [The End of Assistant B's Answer]

    [The Start of RUBRIC]
    {rubric}
    [The End of RUBRIC]

    [The Start of Judge]
    """
).strip()


RUBRIC_AUGMENTED_VERIFICATION_PROMPT = dedent(
    """
    You are given a user question, two responses from two AI assistants, and a specific evaluation rubric.
    Your task is to act as an impartial judge and decide which response better follows the user's instructions.
    You are provided with an evaluation rubric, but you must exercise vigilance. The provided rubric may be flawed, incomplete, or fundamentally incorrect (for example, it might ignore the user's core intent or guide you toward a wrong conclusion).
    You must not treat the provided rubric as absolute truth if it contradicts the user's instructions or logical reasoning.

    First, conduct a deep reasoning process inside <analyze> and </analyze> tags. Your reasoning must follow these steps:
    1. Rubric Validity Check and Ideal Answer Formulation:
       - Analyze the User Question carefully. What is the core intent?
       - Formulate an Ideal Answer in your mind based on the User Question. What must a correct response contain?
       - Evaluate the provided Rubric. Does it align with the User Question and your Ideal Answer?
       - Determine if the rubric is helpful or misleading.
         - If helpful: The rubric correctly captures the user's intent and logic. Use it as is.
         - If misleading: The rubric contains errors, misses key constraints, or leads to incorrect evaluations. You must explicitly discard the flawed parts and define your own correct rubric or criteria based on the User Question.
    2. Step-by-Step Evaluation:
       - Compare Assistant A and Assistant B against the valid criteria.
       - Be explicit: Which assistant matches the Ideal Answer better?
    3. Comparison: Based on the valid criteria, determine which assistant provided the superior response.

    Be explicit in your thought process. Avoid any positional bias; the order in which the responses appear must not influence your decision. Do not let response length or the assistants' names sway your judgment.

    Output Format:
    After your reasoning, output exactly and only the following tags in this order:
    1) <analyze> ... </analyze>
    2) <rubric>helpful</rubric> or <rubric>misleading</rubric>
    3) <answer>A</answer> or <answer>B</answer>

    Guidance for <rubric>:
    - helpful: Choose this if the provided rubric is logical, accurate, and correctly guides the evaluation of the user's prompt.
    - misleading: Choose this if the provided rubric is flawed, off the point, or would lead to selecting the wrong response if followed strictly.

    [User Question]
    {prompt}

    [The Start of Assistant A's Answer]
    {response_a}
    [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer]
    {response_b}
    [The End of Assistant B's Answer]

    [The Start of RUBRIC]
    {rubric}
    [The End of RUBRIC]
    """
).strip()


def build_rubric_generation_prompt(example: PairwiseExample) -> str:
    """Build the rubric generation prompt for one pairwise example."""

    return RUBRIC_GENERATION_PROMPT.format(**_example_fields(example))


def _example_fields(example: PairwiseExample) -> dict[str, str]:
    """Normalize pairwise example text before prompt formatting."""

    return {
        "prompt": example.prompt.strip() or "(no prompt)",
        "response_a": example.response_a.strip() or "(empty response)",
        "response_b": example.response_b.strip() or "(empty response)",
    }


def build_rubric_free_verification_prompt(example: PairwiseExample) -> str:
    """Build the rubric-free verifier prompt for one pairwise example."""

    return RUBRIC_FREE_VERIFICATION_PROMPT.format(**_example_fields(example))


def build_rubric_conditioned_selection_prompt(
    example: PairwiseExample,
    rubric_text: str,
) -> str:
    """Build the rubric-conditioned A/B selection prompt used for margin scoring."""

    return RUBRIC_CONDITIONED_SELECTION_PROMPT.format(
        **_example_fields(example),
        rubric=rubric_text.strip() or "(empty rubric)",
    )


def build_rubric_augmented_verification_prompt(
    example: PairwiseExample,
    rubric_text: str,
) -> str:
    """Build the vigilance-aware verifier prompt used for GRPO and inference."""

    return RUBRIC_AUGMENTED_VERIFICATION_PROMPT.format(
        **_example_fields(example),
        rubric=rubric_text.strip() or "(empty rubric)",
    )
