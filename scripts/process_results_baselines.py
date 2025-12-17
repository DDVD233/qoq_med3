import os
import sys
import json
import argparse
import asyncio
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from dataclasses import dataclass
from typing import Dict, Any, List
import anthropic
from openai import AsyncOpenAI


# Grading instructions
MIMEQA_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answers regarding understanding of a mime performance.
You will be shown a "gold-standard" answer from a human annotator, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to determine whether the candidate captures the core meaning of the reference answer using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate does not contain misleading information and does not hallucinate story plots not present in the reference answer.
3. Since the videos are mime performances, invisible actions, objects, or the mime actor portraying objects should be considered correct if and only if they are relevant to the question.
4. The candidate answer can be a good answer in place of the reference answer as long as they are in the same ballpark. However, the candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

SIQ_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answer regarding understanding of a video.
You will be shown a "gold-standard" answer from human annotators, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to judge whether the candidate captures the core meaning of the reference answer using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate's explanation is semantically equivalent as the reference and does not add a claim that conflicts with it. 
3. The candidate should not assert a conflicting explanation or introduce factually incompatible details. The candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

INTENTQA_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the question and answers about the intent behind actions in a video.
You will be shown a "gold-standard" answer from human annotators, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to judge whether the candidate gives a plausible interpretation of the intent that does not contradict the reference, using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate's explanation is in the same ballpark as the reference and does not add a claim that conflicts with it. The wording need not be the same; minor additions are allowed if they are consistent with the reference and the question.
3. The candidate should not assert a conflicting explanation, introduce factually incompatible details, or miss the core intent. The candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

GRADE_PROMPT = """
Question:
"{question}"
Candidate Answer:
"{candidate_answer}"
Reference Answer:
"{ref_answer}"

Please evaluate the candidate answer based on the dataset-specific instructions.

Respond with exactly this format - a JSON object with two fields:
- "correct": true or false (boolean)
- "explanation": a very short, few phrases explanation of your decision (string)
Only respond with the JSON object, no other text or comments.
"""


@dataclass
class Evaluation:
    correct: bool
    explanation: str
    
    def to_dict(self):
        return {"correct": self.correct, "explanation": self.explanation}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(correct=data["correct"], explanation=data["explanation"])


class JSONParseError(Exception):
    def __init__(self, content: str):
        self.content = content
        super().__init__("JSON parse error")


class LLM:
    def __init__(self, llm_str: str, default_instructions: str | None = None, provider: str = "anthropic"):
        self.llm_str = llm_str
        self.instructions = default_instructions
        self.provider = provider
        
        if provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "openai":
            self.client = AsyncOpenAI(
                api_key=os.getenv("MIT_OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'anthropic', 'openai'")
    
    @retry(reraise=True, wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def create_completion(self, messages: List[Dict[str, str]]) -> Evaluation:
        if self.provider == "anthropic":
            # Convert messages to Claude format
            claude_messages = []
            system_message = self.instructions
            
            for msg in messages:
                if msg["role"] == "user":
                    claude_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
            
            response = await self.client.messages.create(
                model=self.llm_str,
                messages=claude_messages,
                system=system_message,
                max_tokens=2048
            )
            
            original_content = response.content[0].text.strip()
            
        elif self.provider == "openai":
            # Convert messages to OpenAI format
            openai_messages = []
            if self.instructions:
                openai_messages.append({
                    "role": "system",
                    "content": self.instructions
                })
            
            # Add user messages
            for msg in messages:
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Ask OpenAI for structured output matching the Evaluation schema
            response = await self.client.chat.completions.create(
                model=self.llm_str,
                messages=openai_messages,
                max_completion_tokens=2048,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Evaluation",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["correct", "explanation"],
                            "properties": {
                                "correct": {"type": "boolean"},
                                "explanation": {"type": "string"}
                            }
                        },
                        "strict": True
                    }
                }
            )
            
            # Prefer parsed structured output if SDK provides it
            try:
                parsed = response.choices[0].message.parsed  # type: ignore[attr-defined]
                if parsed is not None:
                    return Evaluation.from_dict(parsed)
            except Exception:
                pass
            
            original_content = response.choices[0].message.content.strip()
        
        try:
            # Handle JSON code blocks
            json_content = original_content
            if json_content.startswith("```json"):
                json_content = json_content[7:]
            if json_content.endswith("```"):
                json_content = json_content[:-3]
            
            json_data = json.loads(json_content.strip())
            return Evaluation.from_dict(json_data)
            
        except json.JSONDecodeError:
            # Trigger retry by raising; caller will fallback after retries
            raise JSONParseError(original_content)


def create_message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


async def check_answer(question: str, candidate_answer: str, reference_answer: str, grade_instruction: str, provider: str, model: str) -> Evaluation:
    grader = LLM(llm_str=model, default_instructions=grade_instruction, provider=provider)
    prompt = GRADE_PROMPT.format(question=question, candidate_answer=candidate_answer, ref_answer=reference_answer)
    message = create_message("user", prompt)
    try:
        return await grader.create_completion([message])
    except JSONParseError as e:
        correct = "true" in e.content.lower()
        return Evaluation(correct=correct, explanation="Error parsing JSON, content: " + e.content)
    except RetryError as e:  # Safety net if reraise behavior changes
        exc = None
        try:
            exc = e.last_attempt.exception()  # type: ignore[attr-defined]
        except Exception:
            pass
        if isinstance(exc, JSONParseError):
            correct = "true" in exc.content.lower()
            return Evaluation(correct=correct, explanation="Error parsing JSON, content: " + exc.content)
        raise


async def evaluate_worker(semaphore, question: str, candidate_answer: str, reference_answer: str, grade_instruction: str, provider: str, model: str):
    async with semaphore:
        evaluation = await check_answer(question, candidate_answer, reference_answer, grade_instruction, provider, model)
        await asyncio.sleep(0.5)  # Rate limiting
        return evaluation


def load_annotations():
    annotation_path = Path("/Users/dvd/Downloads/human_behavior_data")

    # Load annotation files
    mimeqa_annotation_path = annotation_path / "mimeqa" / "metadata.csv"
    siq2_annotation_train_path = annotation_path / "siq2" / "qa" / "qa_train.json"
    siq2_annotation_val_path = annotation_path / "siq2" / "qa" / "qa_val.json"
    intentqa_annotation_train_path = annotation_path / "intentqa" / "annotations" / "train.csv"
    intentqa_annotation_val_path = annotation_path / "intentqa" / "annotations" / "val.csv"
    intentqa_annotation_test_path = annotation_path / "intentqa" / "annotations" / "test.csv"

    mimeqa_annotation = pd.read_csv(mimeqa_annotation_path)
    siq2_annotation_train = pd.read_json(siq2_annotation_train_path, lines=True)
    siq2_annotation_val = pd.read_json(siq2_annotation_val_path, lines=True)
    siq2_annotation = pd.concat([siq2_annotation_train, siq2_annotation_val])
    intentqa_annotation_train = pd.read_csv(intentqa_annotation_train_path)
    intentqa_annotation_val = pd.read_csv(intentqa_annotation_val_path)
    intentqa_annotation_test = pd.read_csv(intentqa_annotation_test_path)
    intentqa_annotation = pd.concat([intentqa_annotation_train, intentqa_annotation_val, intentqa_annotation_test])

    # Create mapping dictionaries
    mimeqa_answer_question_map = dict(zip(mimeqa_annotation['reference_answer'].str.lower(), mimeqa_annotation['question']))
    siq2_answer_question_map = dict(zip(siq2_annotation['ans_corr'].str.lower(), siq2_annotation['q']))
    intentqa_answer_question_map = dict(zip(intentqa_annotation.apply(lambda row: row[f"a{row['answer']}"].lower(), axis=1), intentqa_annotation["question"]))

    return {
        'mimeqa': mimeqa_answer_question_map,
        'siq2': siq2_answer_question_map,
        'intentqa': intentqa_answer_question_map
    }


def augment_results_with_questions(results_json, answer_question_maps):
    for row in results_json:
        dataset = row['dataset']
        assert dataset in answer_question_maps, f"Unknown dataset: {dataset}"
        row['gold'] = row['gold'].lower()
        assert row['gold'] in answer_question_maps[dataset], f"Gold answer '{row['gold']}' not found in {dataset} mapping"
        row['question'] = answer_question_maps[dataset][row['gold']]

def extract_answers_if_needed(results_json):
    # if the answer is wrapped in <answer>...</answer>, extract the answer
    for row in results_json:
        if '<answer>' in row['gold'] and '</answer>' in row['gold']:
            row['gold'] = row['gold'].split('<answer>')[1].split('</answer>')[0]



async def evaluate_results(results_json, provider: str, model: str):
    grade_instructions = {
        "mimeqa": MIMEQA_GRADE_INSTRUCTION,
        "siq2": SIQ_GRADE_INSTRUCTION, 
        "intentqa": INTENTQA_GRADE_INSTRUCTION
    }

    max_workers = 50
    semaphore = asyncio.Semaphore(max_workers)
    
    tasks = []
    for row in results_json:
        assert row['dataset'] in grade_instructions, f"Unknown dataset: {row['dataset']}"
        tasks.append(evaluate_worker(
            semaphore, 
            row['question'], 
            row['pred'], 
            row['gold'], 
            grade_instructions[row['dataset']],
            provider,
            model
        ))

    print(f"Evaluating {len(tasks)} results with {provider} ({model})...")
    grade_completions = await tqdm.gather(*tasks, desc="Evaluating results")

    graded_results = []
    for completion, row in zip(grade_completions, results_json):
        graded_results.append({
            "dataset": row['dataset'],
            "pred": row['pred'],
            "gold": row['gold'],
            "question": row.get('question', 'Unknown'),
            "graded_result": completion.correct,
            "graded_result_explanation": completion.explanation,
        })

    return graded_results


def calculate_accuracy(graded_results):
    """Calculate accuracy per dataset."""
    dataset_accuracy = {}
    for row in graded_results:
        if row['dataset'] not in dataset_accuracy:
            dataset_accuracy[row['dataset']] = []
        dataset_accuracy[row['dataset']].append(row['graded_result'])
    
    accuracy_results = {}
    for dataset, results in dataset_accuracy.items():
        accuracy = sum(results) / len(results) if results else 0.0
        accuracy_results[dataset] = accuracy
    
    return accuracy_results


async def main():
    parser = argparse.ArgumentParser(description="Process results and calculate accuracy")
    parser.add_argument("--results_path", help="Path to the results JSON file")
    parser.add_argument("--save_path", help="Optional path to save graded results as JSON")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai", 
                       help="LLM provider to use (default: openai)")
    
    args = parser.parse_args()

    if args.provider == "anthropic":
        model = "claude-3-5-haiku-20241022"
    elif args.provider == "openai":
        model = "gpt-5-nano-2025-08-07"

    # Check if results file exists
    if not os.path.exists(args.results_path):
        print(f"Error: Results file '{args.results_path}' not found")
        sys.exit(1)

    # Check for required environment variables based on provider
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is required when using Anthropic")
        sys.exit(1)
    elif args.provider == "openai" and not os.getenv("MIT_OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required when using OpenAI")
        sys.exit(1)

    # Load results
    print(f"Loading results from {args.results_path}")
    with open(args.results_path, 'r') as f:
        results_json = json.load(f)
    
    print(f"Loaded {len(results_json)} results")

    results_list = []
    for index in range(len(results_json['predictions'])):
        results_list.append({
            'dataset': results_json['datasets'][index],
            'pred': results_json['predictions'][index],
            'gold': results_json['ground_truths'][index]
        })
    results_json = results_list

    # Load annotations and create mappings
    print("Loading annotation data...")
    answer_question_maps = load_annotations()
    
    augment_results_with_questions(results_json, answer_question_maps)
    extract_answers_if_needed(results_json)

    graded_results = await evaluate_results(results_json, args.provider, model)

    accuracy_results = calculate_accuracy(graded_results)
    
    print("\nAccuracy Results:")
    print("-" * 40)
    overall_correct = sum(row['graded_result'] for row in graded_results)
    overall_total = len(graded_results)
    
    for dataset, accuracy in accuracy_results.items():
        dataset_total = len([r for r in graded_results if r['dataset'] == dataset])
        print(f"{dataset}: {accuracy:.4f} ({sum([r['graded_result'] for r in graded_results if r['dataset'] == dataset])}/{dataset_total})")
    
    print(f"\nOverall: {overall_correct/overall_total:.4f} ({overall_correct}/{overall_total})")

    # Save graded results if path provided
    if args.save_path:
        print(f"\nSaving graded results to {args.save_path}")
        with open(args.save_path, 'w') as f:
            json.dump(graded_results, f, indent=2)
        print("Results saved successfully!")
    else:
        save_path = args.results_path.replace(".json", "_metrics.json")
        print(f"\nNo save path provided, saving graded results to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(graded_results, f, indent=2)
    print("Results saved successfully!")


if __name__ == "__main__":
    asyncio.run(main())
