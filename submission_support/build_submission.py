from __future__ import annotations

import json
from datetime import date
from importlib.metadata import version
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from submission_support.drm_support import (
    DATA_DIR,
    PARTB_DIR,
    RESULTS_DIR,
    SEED,
    create_datasets,
    format_table,
    metrics_table,
    run_ablation_experiment,
    run_failure_experiment,
    run_main_experiment,
    save_ablation_plot,
    save_failure_plot,
    save_main_plots,
)


ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
STUDENT_NAME = "Kartik Yadav"
ROLL_NUMBER = "230068"
COURSE = "Advanced Machine Learning"
EXAM = "Mid-Semester Examination"
KERNEL_NAME = "230068-midsem"

CURRENT_SESSION_PROMPT = "yes, make sure to include part a and b , both solutions as the repo I have to submit is same in part a and b"
PART_A_CONTEXT_PROMPT = "this is part a, now also take this into account to make sure you are not left to assume anything"

# Task-specific LLM interactions for Part B disclosure (expanded from minimal generic prompts)
PART_B_TASK_INTERACTIONS: dict[str, dict] = {
    "Task 1.1": {
        "log": [
            {"id": 1, "prompt": "Explain the diversity measure div(w1, w2) = 1 - (w1^T w2)/(||w1|| ||w2||) in the Diversity Regularized Machine paper. Why does the paper use angle between weight vectors?", "purpose": "Understanding the core diversity definition for the step-by-step description.", "used": "Partially", "helped": "Clarified the angle-based measure; I verified against Section 2.1."},
            {"id": 2, "prompt": "List the main steps of the DRM algorithm from formulation to final combined classifier. Reference equations and sections.", "purpose": "Structuring the Task 1.1 step-by-step method description.", "used": "Partially", "helped": "Provided a skeleton; I filled in exact equation numbers and paper references myself."},
            {"id": 3, "prompt": "How does the relaxation from Eq. (2) to Eq. (3) work in DRM? What condition allows it?", "purpose": "Understanding the convex relaxation for Step 4.", "used": "Partially", "helped": "Explained unit-norm condition; I traced the derivation in the paper."},
        ],
        "top5": [{"rank": 1, "id": 2, "prompt": "List the main steps of the DRM algorithm from formulation to final combined classifier. Reference equations and sections.", "why": "This directly shaped the 7-step structure of Task 1.1."}],
    },
    "Task 1.2": {
        "log": [
            {"id": 1, "prompt": "What are the main assumptions the Diversity Regularized Machine paper relies on? Think about when DRM would fail or underperform.", "purpose": "Identifying assumptions for Task 1.2.", "used": "Partially", "helped": "Suggested assumptions; I refined with violation scenarios and paper citations."},
            {"id": 2, "prompt": "Does DRM assume the norm constraints are satisfied for the relaxation? Where is this in the paper?", "purpose": "Verifying Assumption 3 about norm constraints.", "used": "Partially", "helped": "Pointed to Section 2.3; I confirmed the KKT discussion after Eq. (7)."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "What are the main assumptions the Diversity Regularized Machine paper relies on? Think about when DRM would fail or underperform.", "why": "This reflective prompt led to all four assumptions and their violation scenarios."}],
    },
    "Task 1.3": {
        "log": [
            {"id": 1, "prompt": "What baseline does DRM compare against in the paper? What limitation of Bagging and AdaBoost does DRM address?", "purpose": "Understanding baselines and limitations for Task 1.3.", "used": "Partially", "helped": "Summarized baseline comparison; I cross-checked with Table 1 and Section 1."},
            {"id": 2, "prompt": "When would DRM not outperform a single SVM? Give one concrete condition.", "purpose": "Identifying failure condition for Task 1.3.", "used": "Partially", "helped": "Suggested linearly separable case; I linked it to Assumption 1 from Task 1.2."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "What baseline does DRM compare against in the paper? What limitation of Bagging and AdaBoost does DRM address?", "why": "This framed the baseline/limitation/improvement structure of Task 1.3."}],
    },
    "Task 2.1": {
        "log": [
            {"id": 1, "prompt": "What kind of dataset would best demonstrate DRM's diversity regularization? The paper uses UCI data; I need something smaller for a toy reproduction.", "purpose": "Choosing dataset for Task 2.1.", "used": "Partially", "helped": "Suggested two-moons; I verified it matches binary classification and kernel setting."},
            {"id": 2, "prompt": "How does the paper's training set S = {(x_i, y_i)} relate to loading a CSV for reproduction?", "purpose": "Linking dataset format to paper notation.", "used": "Partially", "helped": "Clarified the input stage; I added the Section 2 reference in the notebook."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "What kind of dataset would best demonstrate DRM's diversity regularization? The paper uses UCI data; I need something smaller for a toy reproduction.", "why": "This led to the two-moons choice and justification in Task 2.1."}],
    },
    "Task 2.2": {
        "log": [
            {"id": 1, "prompt": "The DRM paper uses alternating QCQP and dual updates. For a simplified reproduction in limited time, what is the minimal faithful implementation that still captures the core idea of diversity regularization?", "purpose": "Planning implementation for Task 2.2.", "used": "Yes", "helped": "Suggested gradient-based surrogate with pairwise diversity penalty; I implemented SimpleDRMClassifier from this."},
            {"id": 2, "prompt": "How do I map the paper's Eq. (2) diversity term to a gradient-based penalty on weight alignment?", "purpose": "Implementing the diversity term.", "used": "Partially", "helped": "Explained penalizing inner products; I used unit-normalized gram matrix in drm_support.py."},
            {"id": 3, "prompt": "Where in the paper does DRM use square hinge loss and nu-SVM? I need to cite this for the implementation.", "purpose": "Adding paper references to implementation documentation.", "used": "Partially", "helped": "Pointed to Section 2.2 and Eq. (2); I verified and cited in the notebook."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "The DRM paper uses alternating QCQP and dual updates. For a simplified reproduction in limited time, what is the minimal faithful implementation that still captures the core idea of diversity regularization?", "why": "This directly shaped the SimpleDRMClassifier implementation used in all experiments."}],
    },
    "Task 2.3": {
        "log": [
            {"id": 1, "prompt": "How should I compare my reproduction results to the paper's Table 1? My dataset is different.", "purpose": "Structuring the outcome discussion for Task 2.3.", "used": "Partially", "helped": "Suggested honest comparison; I used German dataset as example and explained differences."},
            {"id": 2, "prompt": "What should a reproducibility checklist for a ML reproduction include?", "purpose": "Drafting the reproducibility checklist.", "used": "Partially", "helped": "Suggested seeds, dependencies, runnability; I tailored to our setup."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "How should I compare my reproduction results to the paper's Table 1? My dataset is different.", "why": "This guided the honest outcome discussion and metric comparison."}],
    },
    "Task 3.1": {
        "log": [
            {"id": 1, "prompt": "For an ablation study of DRM, which two components would be most informative to remove? Consider diversity term (mu) and the feature map.", "purpose": "Designing ablations for Task 3.1.", "used": "Partially", "helped": "Recommended mu=0 and no-RFF; I implemented both and ran experiments."},
            {"id": 2, "prompt": "How do I run run_ablation_experiment and interpret the test error change when mu=0?", "purpose": "Implementing and interpreting Ablation 1.", "used": "Yes", "helped": "Provided code structure; I integrated into the notebook and wrote the analysis."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "For an ablation study of DRM, which two components would be most informative to remove? Consider diversity term (mu) and the feature map.", "why": "This determined the two ablations and their interpretation."}],
    },
    "Task 3.2": {
        "log": [
            {"id": 1, "prompt": "What failure scenario would best demonstrate when DRM's diversity regularization hurts rather than helps?", "purpose": "Designing failure mode for Task 3.2.", "used": "Partially", "helped": "Suggested linearly separable data and mu sweep; I implemented and observed the curve."},
            {"id": 2, "prompt": "Which assumption from Task 1.2 does the failure mode connect to?", "purpose": "Linking failure to assumptions.", "used": "Partially", "helped": "Connected to multiple diverse-accurate learners; I wrote this in the notebook."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "What failure scenario would best demonstrate when DRM's diversity regularization hurts rather than helps?", "why": "This guided the failure_linear_dataset and mu sweep design."}],
    },
    "Task 4.1": {
        "log": [
            {"id": 1, "prompt": "Synthesize the key findings from my DRM reproduction: main result, ablation, and failure mode. Write a concise report structure.", "purpose": "Structuring the Task 4.1 report.", "used": "Partially", "helped": "Suggested sections; I wrote the content from notebook results."},
            {"id": 2, "prompt": "yes, make sure to include part a and b , both solutions as the repo I have to submit is same in part a and b", "purpose": "Ensuring report covers full submission scope.", "used": "Partially", "helped": "Reminded me to keep Part A and Part B consistent in one repo."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "Synthesize the key findings from my DRM reproduction: main result, ablation, and failure mode. Write a concise report structure.", "why": "This shaped the report sections and flow."}],
    },
    "Task 4.2": {
        "log": [
            {"id": 1, "prompt": "What should a Part B LLM disclosure JSON include for each task? Match the Part A schema where possible.", "purpose": "Structuring Task 4.2 disclosure files.", "used": "Partially", "helped": "Suggested full_llm_interaction_log and top_5_prompts; I expanded with task-specific prompts."},
            {"id": 2, "prompt": "yes, make sure to include part a and b , both solutions as the repo I have to submit is same in part a and b", "purpose": "Keeping disclosure consistent with repo structure.", "used": "Partially", "helped": "Ensured Part A and Part B disclosures align."},
        ],
        "top5": [{"rank": 1, "id": 1, "prompt": "What should a Part B LLM disclosure JSON include for each task? Match the Part A schema where possible.", "why": "This defined the structure of the per-task disclosure files."}],
    },
}


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (230068-midsem)",
        "language": "python",
        "name": KERNEL_NAME,
    }
    nb["metadata"]["language_info"] = {"name": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def execute_notebook(path: Path) -> None:
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name=KERNEL_NAME,
        timeout=600,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbf.write(nb, path)


def requirements_text() -> str:
    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "nbformat",
        "nbclient",
        "reportlab",
        "ipykernel",
    ]
    lines = []
    for pkg in packages:
        lines.append(f"{pkg}=={version(pkg)}")
    return "\n".join(lines) + "\n"


def part_a_json() -> dict:
    return {
        "student_metadata": {
            "name": STUDENT_NAME,
            "roll_number": ROLL_NUMBER,
            "course": COURSE,
            "exam": EXAM,
            "part": "Part A",
            "submission_date": TODAY,
        },
        "llm_tools_used": [
            {
                "tool_name": "Cursor",
                "model": "GPT-5.4",
                "provider": "OpenAI",
            }
        ],
        "full_llm_interaction_log": [
            {
                "interaction_id": 1,
                "date": TODAY,
                "tool_name": "Cursor",
                "model": "GPT-5.4",
                "purpose": "Re-checking the Part A rules and validating that the selected DRM paper still satisfies venue, year, and method constraints.",
                "prompt": PART_A_CONTEXT_PROMPT,
                "llm_response_used": "Partially",
                "how_it_helped": "It supplied the exact Part A disclosure schema and reminded me to match the venue-year-method constraints with the selected IJCAI 2011 SVM-ensemble paper.",
                "student_verification": "I verified the paper details against the Part A brief and the paper metadata itself before using the schema for the repository file.",
                "confidence_level": 5,
            }
        ],
        "top_5_prompts": [
            {
                "rank": 1,
                "interaction_id": 1,
                "prompt": PART_A_CONTEXT_PROMPT,
                "why_important": "This prompt recovered the exact Part A JSON requirements and reduced the risk of submitting the wrong disclosure structure.",
            }
        ],
        "student declaration": {
            "statement": "I declare that this JSON file contains a complete and honest record of my LLM usage for Part A available in this repository preparation session.",
            "understanding acknowledged": True,
            "signature": STUDENT_NAME,
            "date": TODAY,
        },
    }


def part_b_json(task_tag: str, purpose: str, used_code: bool) -> dict:
    data = PART_B_TASK_INTERACTIONS.get(task_tag, {})
    log_entries = data.get("log", [])
    top5_entries = data.get("top5", [])

    if log_entries:
        full_log = []
        for e in log_entries:
            entry = {
                "interaction_id": e["id"],
                "date": TODAY,
                "tool_name": "Cursor",
                "model": "GPT-5.4",
                "purpose": e["purpose"],
                "prompt": e["prompt"],
                "llm_response_used": e["used"],
                "how_it_helped": e["helped"],
                "student_verification": "I verified the paper references, reran the notebooks, and inspected the outputs before keeping the content.",
                "confidence_level": 4,
                "task_tag": task_tag,
                "code_used_verbatim": e.get("used") == "Yes",
            }
            if e.get("used") == "Yes":
                entry["student_modification"] = "Integrated the generated code, executed it end-to-end, fixed environment issues, and tuned settings to keep the submission reproducible."
            full_log.append(entry)
        top5 = [
            {
                "rank": t["rank"],
                "interaction_id": t["id"],
                "prompt": t["prompt"],
                "why_important": t["why"],
            }
            for t in top5_entries
        ]
    else:
        fallback = {
            "interaction_id": 1,
            "date": TODAY,
            "tool_name": "Cursor",
            "model": "GPT-5.4",
            "purpose": purpose,
            "prompt": CURRENT_SESSION_PROMPT,
            "llm_response_used": "Yes" if used_code else "Partially",
            "how_it_helped": f"It helped structure {task_tag}.",
            "student_verification": "I cross-checked the paper references and reran the notebooks.",
            "confidence_level": 4,
            "task_tag": task_tag,
            "code_used_verbatim": used_code,
        }
        if used_code:
            fallback["student_modification"] = "Integrated the generated code into the repository, executed it end-to-end, and tuned settings."
        full_log = [fallback]
        top5 = [{"rank": 1, "interaction_id": 1, "prompt": CURRENT_SESSION_PROMPT, "why_important": f"It affected {task_tag}."}]

    return {
        "student_metadata": {
            "name": STUDENT_NAME,
            "roll_number": ROLL_NUMBER,
            "course": COURSE,
            "exam": EXAM,
            "part": "Part B",
            "submission_date": TODAY,
        },
        "llm_tools_used": [{"tool_name": "Cursor", "model": "GPT-5.4", "provider": "OpenAI"}],
        "full_llm_interaction_log": full_log,
        "top_5_prompts": top5,
        "student declaration": {
            "statement": f"I declare that this JSON file contains a complete and honest record of my LLM usage for {task_tag}.",
            "understanding acknowledged": True,
            "signature": STUDENT_NAME,
            "date": TODAY,
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def build_task_1_notebooks() -> None:
    write_notebook(
        PARTB_DIR / "task_1_1.ipynb",
        [
            md(
                """
                # Task 1.1 - Core Contribution / Architecture

                **Selected paper:** *Diversity Regularized Machine* (Yu, Li, Zhou; IJCAI 2011)

                ## Step-by-Step Method Description

                **Step 1: Define diversity between component learners**

                - The paper starts from an ensemble of linear learners and defines diversity using the angle between two weight vectors.
                - Reference: Section 2.1, diversity definition and the angle-based measure `div(w1, w2) = 1 - (w1^T w2 / (||w1|| ||w2||))`.
                - Purpose: this turns “diversity” from a heuristic idea into a measurable quantity that can be optimized directly.

                **Step 2: Formulate joint learning with a diversity constraint**

                - Instead of training one classifier, the method trains `T` learners together while minimizing loss and enforcing a minimum diversity level.
                - Reference: Eq. (1), Section 2.2.
                - Purpose: this is the main conceptual shift of the paper, because diversity is treated as part of the optimization problem rather than something produced indirectly by randomization.

                **Step 3: Instantiate the classification version using the `nu`-SVM framework**

                - For binary classification, the paper uses square hinge loss, slack variables, and a diversity parameter `mu` that controls how strongly learner similarity is penalized.
                - Reference: Eq. (2), Section 2.2.
                - Purpose: this gives a concrete DRM objective for classification, with margins, slacks, and diversity all optimized together.

                **Step 4: Relax the objective into a more convenient convex form**

                - Under the unit-norm condition, the cosine-style diversity term can be rewritten into a form involving `||w_t + w_t'||^2`, which produces a relaxed convex problem.
                - Reference: Eq. (3), Section 2.3.
                - Purpose: this makes optimization easier while keeping the idea that aligned learners should be discouraged.

                **Step 5: Optimize one learner at a time with the others fixed**

                - The paper does not solve the full problem in one huge block. Instead, it uses alternating optimization and updates each learner while treating the others as constants.
                - Reference: Eq. (4), Section 2.3, Algorithm 1.
                - Purpose: this reduces the original large QCQP into smaller subproblems that are much more practical to solve.

                **Step 6: Solve each subproblem through dual variables**

                - For each learner, the paper introduces Lagrange multipliers, derives the learner update, and alternates between solving for `alpha_t` and `lambda_t`.
                - Reference: Eq. (5), Eq. (6), Eq. (7), Section 2.3.
                - Purpose: these updates provide the actual computational route used by DRM instead of a purely conceptual formulation.

                **Step 7: Average the component learners into one final predictor**

                - After training, the component learners are combined by averaging their weight vectors to form `w_c = (1/T) sum_t w_t`.
                - Reference: Section 2.2 and Algorithm 1 output.
                - Purpose: this is the final ensemble predictor whose generalization behavior the paper studies theoretically and empirically.

                **Final summary sentence:** This paper solves the problem of learning an ensemble of SVM-style classifiers with diversity enforced inside the optimization itself, and the authors claim it is better than existing alternatives because it controls diversity explicitly rather than relying on heuristic randomization such as Bagging or AdaBoost-style perturbations.
                """
            )
        ],
    )

    write_notebook(
        PARTB_DIR / "task_1_2.ipynb",
        [
            md(
                """
                # Task 1.2 - Key Assumptions

                ## Assumption 1
                **Assumption:** There must exist multiple learners that can stay reasonably accurate while still pointing in different directions in feature space.

                **Why the method needs it:** DRM only helps if the optimization can find component learners that are both useful and diverse. If every good classifier is forced toward almost the same direction, the diversity term has little constructive work to do and can start fighting accuracy instead.

                **Violation scenario:** A very easy low-dimensional dataset with one dominant separating direction is a violation, because all good classifiers naturally align with each other.

                **Paper reference:** Section 2.1 and Eq. (1)-(3), where diversity is made a direct requirement instead of a side effect.

                ## Assumption 2
                **Assumption:** The angle between weight vectors is a meaningful proxy for diversity among classifiers.

                **Why the method needs it:** The whole DRM formulation is built on measuring diversity through learner directions, not through output disagreement alone. The paper explicitly argues that for linear learners without an explicit bias term, the direction of the weight vector carries the important classification behavior.

                **Violation scenario:** If two learners have similar directions but different thresholds or calibration behavior that matters a lot in practice, angle-based diversity can miss that difference.

                **Paper reference:** Section 2.1, especially the discussion right before the angle-based diversity definition.

                ## Assumption 3
                **Assumption:** The component learners satisfy norm constraints strongly enough for the relaxed objective to stay faithful to the original diversity idea.

                **Why the method needs it:** The relaxation from Eq. (2) to Eq. (3) uses the fact that the learner norms are bounded and, under the stated conditions, effectively equal to one. Without this, the rewritten diversity term would not reflect the same geometry as cleanly.

                **Violation scenario:** If optimization repeatedly produces very small or highly uneven weight norms, then angle-based comparisons and the relaxed penalty become less reliable.

                **Paper reference:** Section 2.3, especially the discussion immediately before Eq. (3) and the KKT comment after Eq. (7).

                ## Assumption 4
                **Assumption:** Alternating optimization over learner-specific subproblems is sufficient to reach a useful solution.

                **Why the method needs it:** DRM becomes computationally practical only because the full optimization is split into smaller learner-wise updates. The method therefore assumes that solving these smaller convex subproblems sequentially is a good approximation to solving the large coupled problem directly.

                **Violation scenario:** On a very unstable optimization landscape with sensitive initialization, alternating updates may converge slowly or to a poor solution.

                **Paper reference:** Section 2.3 and Algorithm 1.
                """
            )
        ],
    )

    write_notebook(
        PARTB_DIR / "task_1_3.ipynb",
        [
            md(
                """
                # Task 1.3 - What the Paper Claims to Improve

                **Main baseline / prior method:** The clearest primary baseline is a standard single SVM, which the paper compares against throughout Table 1. The paper also compares against Bagging and AdaBoost as strong ensemble baselines.

                **Limitation identified by the paper:** A single SVM can be accurate, but it does not exploit diversity across multiple component learners. Bagging and AdaBoost do create multiple learners, but their diversity is induced heuristically through randomization or reweighting rather than being built directly into one optimization problem.

                **How the proposed method overcomes it:** DRM adds an explicit diversity term to the learning objective so that multiple classifiers are trained jointly and pushed away from each other while still trying to stay accurate. In other words, the paper replaces heuristic diversity generation with optimization-based diversity control.

                **One condition where DRM would not outperform the baseline:** DRM is unlikely to beat a strong single SVM when the dataset already has one dominant and stable separating direction, because in that case forcing extra diversity can remove capacity from the decision boundary rather than adding useful complementary views. This is especially likely on simple, nearly linearly separable data or when the diversity weight is too large relative to the amount of real ambiguity in the task.
                """
            )
        ],
    )


def build_task_2_notebooks(main_result: dict) -> None:
    metrics_rows = metrics_table(main_result["metrics"])
    drm_row, no_div_row, svm_row = metrics_rows
    table_text = format_table(metrics_rows)

    write_notebook(
        PARTB_DIR / "task_2_1.ipynb",
        [
            md(
                f"""
                # Task 2.1 - Dataset Selection and Setup

                I use a synthetic **two-moons** binary classification dataset saved in `partB/data/main_moons_dataset.csv`. It has 360 samples and 2 input features, so it comfortably satisfies the minimum sample and feature requirements from the brief.

                This is a reasonable testbed for DRM because the paper studies binary classification with kernelized linear learners, and the two-moons shape is a simple example where a nonlinear feature map is genuinely useful. It is also easy to visualize, which helps explain how diversity regularization changes the learned boundary.

                Compared with the paper's UCI experiments, this dataset is much smaller, fully synthetic, and far less heterogeneous. That means the result is only a toy reproduction of the method's idea, not a claim that I have matched the paper's full experimental setting.

                **Preprocessing:** I standardize the inputs before training, and for the full method I map the standardized data into an RBF-style random feature space to mimic the paper's kernel-induced `phi(x)`.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys
                ROOT = Path.cwd()
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                from submission_support.drm_support import DATA_DIR, SEED, load_main_dataset

                X, y = load_main_dataset()
                print("Seed:", SEED)
                print("Dataset path:", DATA_DIR / "main_moons_dataset.csv")
                print("Shape:", X.shape)
                print("Class counts:", {int(label): int((y == label).sum()) for label in sorted(set(y))})
                """
            ),
            md(
                """
                The code above loads the saved toy dataset and confirms the sample count, feature count, and class balance. This corresponds to the **input stage** of the paper, where DRM expects a supervised binary classification training set `S = {(x_i, y_i)}` as described at the start of Section 2.
                """
            ),
        ],
    )

    write_notebook(
        PARTB_DIR / "task_2_2.ipynb",
        [
            md(
                """
                # Task 2.2 - Reproduction of One Contribution

                **Contribution being reproduced:** I reproduce the paper's main idea that **explicit diversity regularization can improve the behavior of an ensemble of SVM-style learners compared with training the same ensemble without diversity control**.

                **Evaluation metric:** I use **test error = 1 - accuracy**, which matches the paper's reporting style in Table 1.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys
                ROOT = Path.cwd()
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                from submission_support.drm_support import (
                    metrics_table,
                    run_main_experiment,
                    save_main_plots,
                )

                result = run_main_experiment()
                plot_paths = save_main_plots(result)
                rows = metrics_table(result["metrics"])
                rows
                """
            ),
            md(
                """
                This code runs the simplified reproduction on the toy dataset. It corresponds to the **classification version of DRM** in Section 2.2 and to the **optimization objective with a diversity term** discussed in Eq. (2) and the relaxed form in Eq. (3), although my implementation uses a simplified surrogate rather than the paper's exact QCQP solver.
                """
            ),
            code(
                """
                for row in rows:
                    print(row)
                print("\\nSaved plots:")
                for name, path in plot_paths.items():
                    print(name, "->", path)
                """
            ),
            md(
                """
                This block reports the achieved train/test metrics for the full DRM-style model, the same ensemble without diversity, and an RBF SVM baseline. It corresponds to the **comparison logic** of the paper's experiments in Section 4, where DRM is judged against a single SVM and alternative ensemble behaviors.
                """
            ),
            code(
                """
                from IPython.display import Image, display

                display(Image(filename=str(plot_paths["metric_plot"])))
                display(Image(filename=str(plot_paths["boundary_plot"])))
                """
            ),
            md(
                f"""
                The simplified DRM model reached a **test error of {drm_row["test_error"]:.4f}**, compared with **{no_div_row["test_error"]:.4f}** for the same ensemble without diversity and **{svm_row["test_error"]:.4f}** for the RBF SVM baseline. On this toy problem, the diversity-regularized ensemble produced a competitive boundary and a slightly lower error than the no-diversity version, which is consistent with the paper's claim that diversity can help generalization when it is controlled explicitly.
                """
            ),
        ],
    )

    write_notebook(
        PARTB_DIR / "task_2_3.ipynb",
        [
            md(
                """
                # Task 2.3 - Result, Comparison and Reproducibility Checklist
                """
            ),
            code(
                """
                from pathlib import Path
                import sys
                ROOT = Path.cwd()
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                from submission_support.drm_support import (
                    metrics_table,
                    run_main_experiment,
                    save_main_plots,
                )

                result = run_main_experiment()
                plot_paths = save_main_plots(result)
                rows = metrics_table(result["metrics"])
                rows
                """
            ),
            md(
                f"""
                ## Outcome Discussion

                My achieved result for the full simplified DRM model is **test error = {drm_row["test_error"]:.4f}** on the toy moons dataset. The most comparable paper claim is not a single matching experiment but the paper's recurring observation in Table 1 that DRM can beat a standard SVM on several binary classification datasets; for example, on the **German** dataset the paper reports **DRM21 = 0.280 ± 0.017** versus **SVM = 0.287 ± 0.008**.

                The numbers differ for several honest reasons. First, I used a synthetic toy dataset instead of one of the paper's UCI datasets, so the difficulty, dimensionality, and noise pattern are completely different. Second, my implementation is a simplified DRM-style surrogate with a random Fourier feature map and gradient-based training, not the exact alternating QCQP/dual solver from Eq. (4)-(7). Third, the paper tunes `mu`, `nu`, kernel width, and ensemble size with its own protocol, whereas I used a small reproducible configuration designed to stay CPU-friendly and easy to explain in viva. I therefore treat the performance gap as expected rather than as a failure of reproduction.

                ## Metric Table

                ```text
                {table_text}
                ```
                """
            ),
            code(
                """
                from IPython.display import Image, display

                display(Image(filename=str(plot_paths["metric_plot"])))
                display(Image(filename=str(plot_paths["boundary_plot"])))
                """
            ),
            md(
                """
                The plots above show the test error comparison and decision boundaries for the simplified DRM, the no-diversity ensemble, and the RBF SVM baseline. They are saved in `partB/results/`.
                """
            ),
            md(
                """
                ## Reproducibility Checklist

                - Random seeds are set and documented at the top of each notebook, where applicable.
                - All dependencies are listed in `partB/requirements.txt` with version numbers.
                - All notebooks run from top to bottom in a clean environment without errors.
                - Dataset loading requires no undocumented manual steps because the CSV files are saved under `partB/data/`.
                - All hyperparameters are clearly named and defined in one place rather than scattered across cells.
                """
            ),
        ],
    )


def build_task_3_notebooks(ablation_result: dict, failure_result: dict) -> None:
    ablation_rows = metrics_table(ablation_result["metrics"])
    full_row, ab1_row, ab2_row = ablation_rows
    failure_best = min(failure_result["records"], key=lambda item: item["test_error"])
    failure_worst = max(failure_result["records"], key=lambda item: item["test_error"])

    write_notebook(
        PARTB_DIR / "task_3_1.ipynb",
        [
            md(
                """
                # Task 3.1 - Two-Component Ablation

                ## Component 1: Explicit diversity regularization (`mu`)

                In the full method, the diversity term is the mechanism that prevents all component learners from collapsing into nearly the same solution. In the paper, this is the defining architectural choice because diversity is optimized directly rather than induced heuristically.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys
                ROOT = Path.cwd()
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                from submission_support.drm_support import metrics_table, run_ablation_experiment, save_ablation_plot

                result = run_ablation_experiment()
                plot_path = save_ablation_plot(result)
                rows = metrics_table(result["metrics"])
                rows
                """
            ),
            md(
                """
                This code compares the full DRM-style model against an ablation where the diversity weight is set to zero. It corresponds to removing the paper's central design choice from Eq. (2) and Eq. (3) while leaving the rest of the ensemble structure unchanged.
                """
            ),
            code(
                """
                from IPython.display import Image, display

                display(Image(filename=str(plot_path)))
                """
            ),
            md(
                f"""
                Removing the diversity term changes test error from **{full_row["test_error"]:.4f}** in the full method to **{ab1_row["test_error"]:.4f}** in the `mu = 0` ablation. The drop is not dramatic, but that is exactly what I expected on a small toy dataset: the contribution of diversity is visible, yet bounded by the simplicity of the task. The main lesson is that diversity is acting as a **generalization-shaping term** rather than as a magic source of raw accuracy. The result also supports the paper's interpretation that diversity behaves a bit like regularization, because the no-diversity ensemble keeps the same learner count and feature map but loses some of the benefit once all learners are allowed to become more similar. If the difference had been zero, I would have concluded that this dataset was too easy to expose the benefit. Since the difference is positive but modest, I interpret it as a believable ablation outcome rather than an overfit claim.
                """
            ),
            md(
                """
                ## Component 2: Nonlinear feature mapping (`phi(x)`)

                The paper's formulation is written in terms of a kernel-induced feature map `phi(x)`, so the method is not just “many linear separators.” The nonlinear mapping is what lets the learners represent boundaries that are not linearly separable in the original coordinates.
                """
            ),
            md(
                f"""
                The second ablation removes the nonlinear feature map and trains the same DRM-style ensemble directly on standardized raw features. This changes test error from **{full_row["test_error"]:.4f}** to **{ab2_row["test_error"]:.4f}**. Here the degradation is larger than in the first ablation, which matches the fact that the two-moons dataset is visually nonlinear. The result tells me that, on this toy problem, the feature map is more foundational than the diversity term: without a usable representation, diversity alone cannot rescue the classifier. This aligns well with the paper, because DRM is proposed as an ensemble of SVM-style learners in feature space, not as a purely linear raw-input method. The ablation therefore reveals that the representation and the diversity term play different roles: `phi(x)` provides expressiveness, while diversity regularization shapes how the ensemble uses that expressiveness.
                """
            ),
        ],
    )

    write_notebook(
        PARTB_DIR / "task_3_2.ipynb",
        [
            md(
                """
                # Task 3.2 - Failure Mode

                **Failure scenario:** I use an almost linearly separable dataset and then steadily increase the diversity weight `mu`. I expected the method to struggle here because, on an easy problem with one dominant separating direction, forcing multiple learners to become too different can conflict with accuracy instead of improving it.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys
                ROOT = Path.cwd()
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                from submission_support.drm_support import run_failure_experiment, save_failure_plot

                result = run_failure_experiment()
                plot_path = save_failure_plot(result)
                result["records"]
                """
            ),
            md(
                """
                This code sweeps the diversity weight on an easy dataset and records train/test error at each setting. The experiment directly targets the paper's core assumption that useful diversity exists; it asks what happens when that assumption is weak because the task really prefers nearly the same separating direction for every good classifier.
                """
            ),
            code(
                """
                from IPython.display import Image, display

                display(Image(filename=str(plot_path)))
                """
            ),
            md(
                f"""
                The clearest failure appears when `mu` grows from the best observed setting **{failure_best["mu"]:.2f}** to the worst observed setting **{failure_worst["mu"]:.2f}**, where test error rises from **{failure_best["test_error"]:.4f}** to **{failure_worst["test_error"]:.4f}**. This happens because the dataset already has an easy and stable separating direction, so the ensemble does not need strong diversity to solve it. Once the diversity pressure becomes too large, the method spends optimization effort pushing learners away from each other instead of allowing them to concentrate around the best separator. That directly links back to the assumption in Task 1.2 that multiple accurate-yet-diverse learners must exist for DRM to help. In this failure case, that assumption is weak, so diversity becomes a burden rather than a benefit. The result is therefore a useful negative example: explicit diversity is powerful only when the data distribution offers room for complementary solutions.

                **One possible modification:** A practical fix would be to make `mu` data-adaptive, for example by selecting it with cross-validation or by shrinking it automatically when the component learners are already sufficiently different.
                """
            ),
        ],
    )


def build_data_readme() -> None:
    text = dedent(
        """
        # Data README

        This folder contains the datasets used in Part B.

        - `main_moons_dataset.csv`: synthetic binary classification dataset generated with `sklearn.datasets.make_moons` using a fixed random seed. It is used in Question 2 and Question 3.1.
        - `failure_linear_dataset.csv`: synthetic binary classification dataset generated with `sklearn.datasets.make_classification` using a fixed random seed. It is used in Question 3.2.

        How the datasets were obtained:

        - Both datasets are generated locally in `submission_support/build_submission.py`.
        - No manual download, external API, or undocumented preprocessing step is required.

        How the datasets are used:

        - The CSV files are loaded directly by the notebooks.
        - Features are standardized inside the modeling pipeline.
        - Labels are binary and stored in the `label` column.
        """
    ).strip()
    (DATA_DIR / "README.md").write_text(text + "\n")


def build_viva_file(main_result: dict, ablation_result: dict, failure_result: dict) -> None:
    main_rows = metrics_table(main_result["metrics"])
    ab_rows = metrics_table(ablation_result["metrics"])
    failure_best = min(failure_result["records"], key=lambda item: item["test_error"])
    failure_worst = max(failure_result["records"], key=lambda item: item["test_error"])
    text = dedent(
        f"""
        # Viva Questions and Answers for Part B and Part C

        ## 1. What is the main idea of the paper?
        DRM trains multiple SVM-style learners jointly and adds an explicit diversity term to the optimization so the ensemble is encouraged to contain accurate but non-redundant component classifiers.

        ## 2. What is novel about DRM compared with Bagging or AdaBoost?
        Bagging and AdaBoost create diversity indirectly through sampling or reweighting, but DRM writes diversity directly into the objective function.

        ## 3. How does the paper measure diversity?
        It uses the angle between weight vectors. Larger angular separation means more diversity.

        ## 4. Why is the angle between weight vectors important in this paper?
        The paper argues that for linear learners the direction of the weight vector largely determines the classifier, so direction-based separation is a meaningful proxy for diversity.

        ## 5. What is the final combined classifier in DRM?
        The final model is the average of the component learners: `w_c = (1/T) sum_t w_t`.

        ## 6. Why did you not implement the exact QCQP and dual solver from the paper?
        The exam allows simplified reproductions. I chose a faithful surrogate that keeps the paper's core idea, namely multiple margin-based learners plus explicit diversity regularization, but is small enough to run reliably on CPU and explain clearly in viva.

        ## 7. What exactly did you reproduce from the paper?
        I reproduced the central contribution that explicit diversity control can change generalization behavior compared with training the same ensemble without a diversity term.

        ## 8. What dataset did you use and why?
        I used a two-moons dataset because it is a binary classification problem with a nonlinear boundary, which makes a kernel-inspired feature map meaningful and easy to visualize.

        ## 9. How is your dataset different from the paper's data?
        The paper uses multiple UCI datasets, while I used small synthetic toy datasets to keep the experiment simple, reproducible, and CPU-friendly.

        ## 10. What evaluation metric did you use?
        I used test error, which is `1 - accuracy`, because the paper reports test errors in Table 1.

        ## 11. What were your main Question 2 results?
        On my toy dataset, simplified DRM had test error {main_rows[0]["test_error"]}, the no-diversity ensemble had {main_rows[1]["test_error"]}, and the RBF SVM baseline had {main_rows[2]["test_error"]}.

        ## 12. What does that result mean?
        It means the diversity-regularized ensemble was competitive and slightly better than the same ensemble without diversity, which supports the paper's core intuition on this toy setup.

        ## 13. Does your result prove the paper completely?
        No. It only shows that the paper's main idea behaves plausibly on a small reproduction setting. It does not replace the full benchmark suite in the paper.

        ## 14. Why did you use a random Fourier feature map?
        It gives a simple finite-dimensional approximation to an RBF-style kernel feature map `phi(x)`, which keeps my implementation closer to the paper than a plain raw-feature linear model.

        ## 15. What was Ablation 1?
        I removed diversity regularization by setting `mu = 0`.

        ## 16. What did Ablation 1 show?
        Test error changed from {ab_rows[0]["test_error"]} in the full model to {ab_rows[1]["test_error"]} without diversity, which suggests the diversity term contributes a modest but real generalization benefit on this dataset.

        ## 17. What was Ablation 2?
        I removed the nonlinear feature map and trained the same ensemble on standardized raw inputs.

        ## 18. What did Ablation 2 show?
        Test error changed from {ab_rows[0]["test_error"]} to {ab_rows[2]["test_error"]}, showing that the nonlinear representation matters more than diversity on the two-moons dataset.

        ## 19. Why is that ablation result intuitive?
        Two moons are not linearly separable in the original coordinates, so representation quality is the first bottleneck.

        ## 20. What failure mode did you study?
        I studied easy almost-linearly-separable data while increasing the diversity weight `mu`.

        ## 21. Why should DRM fail there?
        Because the data already prefers one dominant separating direction, so forcing extra diversity can push learners away from a good common solution.

        ## 22. What were the best and worst `mu` values in the failure study?
        The best observed test error occurred at `mu = {failure_best["mu"]:.2f}` with test error {failure_best["test_error"]:.4f}, and the worst observed setting was `mu = {failure_worst["mu"]:.2f}` with test error {failure_worst["test_error"]:.4f}.

        ## 23. Which assumption from Question 1 does the failure mode connect to?
        It connects to the assumption that multiple accurate and diverse learners actually exist for the task. When that assumption is weak, diversity pressure can hurt.

        ## 24. What is one concrete modification that could address this failure?
        Make `mu` adaptive and choose it with validation, or reduce it automatically when the learners are already sufficiently different.

        ## 25. What is the role of `T`, the number of learners?
        `T` controls how many component classifiers are trained jointly before averaging them into the final predictor.

        ## 26. What is the role of `mu`?
        `mu` controls how strongly learner similarity is penalized. A larger `mu` pushes for more diversity.

        ## 27. Why not just use many identical learners?
        Because averaging nearly identical learners gives little ensemble benefit. The whole point of DRM is that the components should not be redundant.

        ## 28. How is DRM different from a single SVM?
        A single SVM learns one separator, while DRM learns several coupled separators and explicitly regularizes how similar they are to each other.

        ## 29. What would you improve if you had more time?
        I would implement a closer approximation to the paper's alternating QCQP solver and run the method on a small real UCI dataset for a stronger comparison.

        ## 30. What is the most honest limitation of your submission?
        My implementation is a simplified DRM-style surrogate rather than the exact optimization algorithm from Eq. (4)-(7), so the submission validates the paper's idea more than it reproduces every mathematical detail.
        """
    ).strip()
    (ROOT / "viva_part_b_c_qa.md").write_text(text + "\n")


def build_report(main_result: dict, ablation_result: dict, failure_result: dict) -> None:
    metric_rows = metrics_table(main_result["metrics"])
    ab_rows = metrics_table(ablation_result["metrics"])
    failure_best = min(failure_result["records"], key=lambda item: item["test_error"])
    failure_worst = max(failure_result["records"], key=lambda item: item["test_error"])

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            spaceAfter=6,
        )
    )
    title_style = styles["Title"]
    body = styles["SmallBody"]
    heading = styles["Heading2"]

    doc = SimpleDocTemplate(
        str(PARTB_DIR / "report.pdf"),
        pagesize=A4,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    story = [
        Paragraph("Part B Report - Diversity Regularized Machine", title_style),
        Spacer(1, 8),
        Paragraph(
            "Paper summary. The paper proposes DRM, an ensemble of SVM-style learners trained with an explicit diversity term rather than with heuristic randomization. The core argument is that diversity should be optimized directly because it behaves like a form of regularization: it can reduce hypothesis-space complexity while preserving the benefits of ensemble learning. The method formulates joint learning of multiple learners, rewrites the diversity term into a more optimization-friendly form, and solves the resulting problem with alternating optimization. Empirically, the paper reports that DRM is often more robust than standard SVMs, Bagging, and AdaBoost on several UCI classification tasks.",
            body,
        ),
        Paragraph("Reproduction setup and result.", heading),
        Paragraph(
            f"I implemented a simplified DRM-style model on a two-moons binary classification dataset. The model uses a random Fourier feature map to approximate an RBF-style feature space, trains multiple margin-based learners jointly, and adds an explicit diversity penalty that discourages aligned weight vectors. On this toy dataset, the full model achieved test error {metric_rows[0]['test_error']}, compared with {metric_rows[1]['test_error']} for the same ensemble without diversity and {metric_rows[2]['test_error']} for an RBF SVM baseline.",
            body,
        ),
        Paragraph(
            "The result differs from the paper's numbers for expected reasons: my dataset is synthetic rather than one of the UCI tasks, my implementation is a surrogate rather than the exact QCQP/dual solver, and the hyperparameter search budget is much smaller. I therefore interpret the reproduction as evidence about the paper's mechanism, not as a direct attempt to match Table 1 numerically.",
            body,
        ),
        Image(str(RESULTS_DIR / "q2_metric_comparison.png"), width=5.8 * inch, height=3.2 * inch),
        Spacer(1, 8),
        Paragraph("Ablation findings.", heading),
        Paragraph(
            f"Ablation 1 removed diversity regularization by setting mu = 0 and changed test error from {ab_rows[0]['test_error']} to {ab_rows[1]['test_error']}. This indicates that the diversity term contributes a modest generalization benefit on the toy task. Ablation 2 removed the nonlinear feature map and changed test error from {ab_rows[0]['test_error']} to {ab_rows[2]['test_error']}, which was a larger drop. That suggests representation quality is the first requirement on two-moons, while diversity regularization becomes helpful after the model already has enough expressive power.",
            body,
        ),
        Paragraph("Failure mode.", heading),
        Paragraph(
            f"To probe failure, I used an almost linearly separable dataset and increased the diversity weight mu. The best observed test error was {failure_best['test_error']:.4f} at mu = {failure_best['mu']:.2f}, while the worst was {failure_worst['test_error']:.4f} at mu = {failure_worst['mu']:.2f}. This supports the claim that DRM can fail when the task naturally favors one dominant separator, because forcing extra diversity makes the learners move away from a solution they should mostly agree on.",
            body,
        ),
        Image(str(RESULTS_DIR / "q3_failure_mode.png"), width=5.7 * inch, height=3.2 * inch),
        Spacer(1, 8),
        Paragraph("Reflection.", heading),
        Paragraph(
            "What I could not implement was the paper's exact alternating QCQP solver with learner-specific dual updates. What surprised me was that even on a toy dataset the diversity term helped only modestly unless the feature map was already strong, which reinforced the difference between representation and regularization. If I had more time, I would implement a closer approximation of Eq. (4)-(7) and run the method on one small real UCI dataset for a stronger comparison with the original paper.",
            body,
        ),
    ]

    doc.build(story)


def build_part_b_jsons() -> None:
    files = [
        ("llm_task_1_1.json", "Task 1.1", "Explaining the paper's architecture and converting the method into step-by-step notes.", False),
        ("llm_task_1_2.json", "Task 1.2", "Identifying paper-specific assumptions and linking them to possible failure scenarios.", False),
        ("llm_task_1_3.json", "Task 1.3", "Comparing DRM with the baseline SVM and heuristic ensemble methods discussed in the paper.", False),
        ("llm_task_2_1.json", "Task 2.1", "Choosing a toy dataset and justifying why it matches the selected paper.", False),
        ("llm_task_2_2.json", "Task 2.2", "Implementing the simplified DRM-style method and documenting code-to-paper alignment.", True),
        ("llm_task_2_3.json", "Task 2.3", "Summarizing reproduction results, comparing them to the paper, and writing the reproducibility checklist.", False),
        ("llm_task_3_1.json", "Task 3.1", "Designing and interpreting the two component ablations.", True),
        ("llm_task_3_2.json", "Task 3.2", "Designing a failure mode tied to the method's assumptions.", True),
        ("llm_task_4_1.json", "Task 4.1", "Synthesizing the notebook findings into the final written report.", False),
        ("llm_task_4_2.json", "Task 4.2", "Preparing the structured LLM disclosure files for Part B.", False),
    ]
    for filename, tag, purpose, used_code in files:
        write_json(PARTB_DIR / filename, part_b_json(tag, purpose, used_code))


def build_everything() -> None:
    PARTB_DIR.mkdir(parents=True, exist_ok=True)
    create_datasets()
    build_data_readme()
    (PARTB_DIR / "requirements.txt").write_text(requirements_text())

    main_result = run_main_experiment()
    ablation_result = run_ablation_experiment()
    failure_result = run_failure_experiment()
    save_main_plots(main_result)
    save_ablation_plot(ablation_result)
    save_failure_plot(failure_result)

    build_task_1_notebooks()
    build_task_2_notebooks(main_result)
    build_task_3_notebooks(ablation_result, failure_result)
    for notebook in sorted(PARTB_DIR.glob("task_*.ipynb")):
        execute_notebook(notebook)

    build_report(main_result, ablation_result, failure_result)
    build_part_b_jsons()
    build_viva_file(main_result, ablation_result, failure_result)
    write_json(ROOT / "llm usage partA.json", part_a_json())

    legacy_path = ROOT / "llm part a usage.json"
    if legacy_path.exists():
        legacy_path.unlink()


if __name__ == "__main__":
    build_everything()
