#!/usr/bin/env python3
"""
Test closed models (GPT-4, Claude, etc.) on medical diagnosis dataset
and evaluate results using the existing evaluation framework.
"""

import json
import os
import sys
import time
import base64
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging
from tqdm import tqdm
import backoff
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import evaluation functions
from examples.reward_function.evaluation import compute_metrics_by_data_source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing different API clients
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available. Install with: pip install anthropic")


class ModelAPIWrapper:
    """Wrapper for different model APIs"""

    def __init__(self, model_type: str, api_key: str, model_name: str,
                 max_retries: int = 3, timeout: int = 60):
        self.model_type = model_type.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout

        if self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed")
            self.client = OpenAI(api_key=api_key)
        elif self.model_type == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not installed")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=3,
        max_time=120
    )
    def call_api(self, prompt: str, images: List[str] = None) -> str:
        """Call the appropriate API with retry logic"""

        if self.model_type == "openai":
            return self._call_openai(prompt, images)
        elif self.model_type == "anthropic":
            return self._call_anthropic(prompt, images)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _call_openai(self, prompt: str, images: List[str] = None) -> str:
        """Call OpenAI API"""
        messages = []

        if images:
            # Create message with images
            content = [{"type": "text", "text": prompt}]
            for img_path in images:
                base64_image = self.encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,  # Use low temperature for consistent medical diagnosis
            max_tokens=500,
            timeout=self.timeout
        )

        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, images: List[str] = None) -> str:
        """Call Anthropic API"""
        messages = []

        if images:
            # Create message with images
            content = []
            for img_path in images:
                base64_image = self.encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            timeout=self.timeout
        )

        return response.content[0].text


class DatasetProcessor:
    """Process medical diagnosis dataset"""

    def __init__(self, dataset_path: str, image_base_path: str = None):
        self.dataset_path = dataset_path
        self.image_base_path = image_base_path or os.path.dirname(dataset_path)
        self.data = self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load JSONL dataset"""
        data = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} examples from {self.dataset_path}")
        return data

    def get_image_paths(self, example: Dict[str, Any]) -> List[str]:
        """Get full image paths for an example"""
        image_paths = []
        for img_path in example.get("images", []):
            full_path = os.path.join(self.image_base_path, img_path)
            if os.path.exists(full_path):
                image_paths.append(full_path)
            else:
                logger.warning(f"Image not found: {full_path}")
        return image_paths

    def extract_demographics(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract demographics from example if available"""
        # Check for demographics in various fields
        demo = example.get("demographics", None)
        if demo:
            return demo

        # Try to extract from problem text
        problem_text = example.get("problem", "").lower()
        demographics = []

        # Check for age
        import re
        age_match = re.search(r'\b(\d{1,3})[- ]?year[- ]?old\b', problem_text)
        if age_match:
            demographics.append(f"age: {age_match.group(1)}")

        # Check for gender
        if "female" in problem_text or "woman" in problem_text:
            demographics.append("gender: female")
        elif "male" in problem_text or "man" in problem_text:
            demographics.append("gender: male")

        return ", ".join(demographics) if demographics else None


class ModelTester:
    """Test models on medical diagnosis dataset"""

    def __init__(self, model_wrapper: ModelAPIWrapper, dataset_processor: DatasetProcessor,
                 output_dir: str = "outputs", batch_size: int = 1, max_workers: int = 1):
        self.model = model_wrapper
        self.dataset = dataset_processor
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results storage
        self.results = {
            "metadata": {
                "model_type": model_wrapper.model_type,
                "model_name": model_wrapper.model_name,
                "dataset_path": dataset_processor.dataset_path,
                "timestamp": datetime.now().isoformat(),
                "total_examples": len(dataset_processor.data)
            },
            "predictions": [],
            "ground_truths": [],
            "data_sources": [],
            "datasets": [],
            "demographics": [],
            "raw_responses": [],
            "examples": []
        }

        # Create checkpoint file path
        self.checkpoint_file = os.path.join(
            self.output_dir,
            f"checkpoint_{model_wrapper.model_type}_{model_wrapper.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Load checkpoint if exists
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            logger.info(f"Loading checkpoint from {self.checkpoint_file}")
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.results = checkpoint
                logger.info(f"Resumed from {len(self.results['predictions'])} completed examples")

    def save_checkpoint(self):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def process_example(self, example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a single example"""

        # Check if already processed (for resuming)
        if idx < len(self.results['predictions']):
            logger.info(f"Skipping example {idx} (already processed)")
            return None

        try:
            # Get image paths
            image_paths = self.dataset.get_image_paths(example)

            # Prepare prompt
            prompt = example['problem']

            # Add instruction for clear answer format
            prompt += "\n\nIMPORTANT: Provide your final answer in the format: \\boxed{answer}"

            # Call API
            start_time = time.time()
            response = self.model.call_api(prompt, image_paths)
            api_time = time.time() - start_time

            # Extract answer from response
            answer = self.extract_answer(response)

            # Get demographics
            demographics = self.dataset.extract_demographics(example)

            result = {
                "idx": idx,
                "prediction": answer,
                "raw_response": response,
                "ground_truth": example['answer'],
                "data_source": example.get('data_source', 'unknown'),
                "dataset": example.get('dataset', 'unknown'),
                "demographics": demographics,
                "api_time": api_time,
                "has_images": len(image_paths) > 0,
                "num_images": len(image_paths)
            }

            logger.info(f"Processed example {idx}: GT='{example['answer']}', Pred='{answer}'")

            return result

        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            return {
                "idx": idx,
                "prediction": "Error",
                "raw_response": str(e),
                "ground_truth": example['answer'],
                "data_source": example.get('data_source', 'unknown'),
                "dataset": example.get('dataset', 'unknown'),
                "demographics": None,
                "api_time": 0,
                "error": str(e)
            }

    def extract_answer(self, response: str) -> str:
        """Extract answer from model response"""
        # Look for boxed answer
        import re
        boxed_match = re.search(r'\\boxed{([^}]*)}', response)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Look for common answer patterns
        response_lower = response.lower()

        # For PE diagnosis
        if "no pe" in response_lower:
            return "No PE"
        elif "chronic pe" in response_lower:
            return "Chronic PE"
        elif "acute pe" in response_lower:
            return "Acute PE"

        # For yes/no questions
        if response_lower.strip().startswith("yes"):
            return "Yes"
        elif response_lower.strip().startswith("no"):
            return "No"

        # Try to extract from "Answer:" pattern
        answer_match = re.search(r'answer[:\s]+([^\n.]+)', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        # Return first line as fallback
        return response.split('\n')[0].strip()

    def run_test(self, num_examples: int = None):
        """Run test on dataset"""

        examples_to_process = self.dataset.data[:num_examples] if num_examples else self.dataset.data

        # Calculate starting index based on checkpoint
        start_idx = len(self.results['predictions'])

        logger.info(f"Testing {len(examples_to_process)} examples starting from index {start_idx}")

        # Process examples with progress bar
        with tqdm(total=len(examples_to_process), initial=start_idx) as pbar:
            for idx, example in enumerate(examples_to_process):
                if idx < start_idx:
                    continue

                result = self.process_example(example, idx)

                if result:
                    # Add to results
                    self.results['predictions'].append(result['prediction'])
                    self.results['ground_truths'].append(result['ground_truth'])
                    self.results['data_sources'].append(result['data_source'])
                    self.results['datasets'].append(result['dataset'])
                    self.results['demographics'].append(result['demographics'])
                    self.results['raw_responses'].append(result['raw_response'])
                    self.results['examples'].append({
                        "idx": idx,
                        "problem": example['problem'],
                        "ground_truth": result['ground_truth'],
                        "prediction": result['prediction'],
                        "raw_response": result['raw_response'],
                        "api_time": result.get('api_time', 0),
                        "error": result.get('error', None)
                    })

                    # Save checkpoint every 10 examples
                    if len(self.results['predictions']) % 10 == 0:
                        self.save_checkpoint()
                        logger.info(f"Checkpoint saved at {len(self.results['predictions'])} examples")

                pbar.update(1)

                # Rate limiting
                time.sleep(0.5)  # Adjust based on API rate limits

        # Final save
        self.save_checkpoint()
        self.save_final_results()

    def save_final_results(self):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full results
        full_results_file = os.path.join(
            self.output_dir,
            f"results_{self.model.model_type}_{self.model.model_name.replace('/', '_')}_{timestamp}.json"
        )
        with open(full_results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Full results saved to {full_results_file}")

        # Save evaluation input format
        eval_input_file = os.path.join(
            self.output_dir,
            f"eval_input_{self.model.model_type}_{self.model.model_name.replace('/', '_')}_{timestamp}.json"
        )
        eval_input = {
            "predictions": self.results['predictions'],
            "ground_truths": self.results['ground_truths'],
            "data_sources": self.results['data_sources'],
            "datasets": self.results['datasets'],
            "demographics": self.results['demographics']
        }
        with open(eval_input_file, 'w') as f:
            json.dump(eval_input, f, indent=2)
        logger.info(f"Evaluation input saved to {eval_input_file}")

        return eval_input_file

    def run_evaluation(self):
        """Run evaluation using the existing evaluation functions"""
        logger.info("Running evaluation...")

        metrics = compute_metrics_by_data_source(
            predictions=self.results['predictions'],
            ground_truths=self.results['ground_truths'],
            data_sources=self.results['data_sources'],
            datasets=self.results['datasets'],
            demographics=self.results['demographics']
        )

        # Save metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(
            self.output_dir,
            f"metrics_{self.model.model_type}_{self.model.model_name.replace('/', '_')}_{timestamp}.json"
        )
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        # Print summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        # Overall metrics
        print("\nOverall Metrics:")
        for key in ["val/accuracy", "val/f1", "val/precision", "val/recall"]:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

        # Fairness metrics
        print("\nFairness Metrics:")
        for key in metrics:
            if key.startswith("overall/"):
                print(f"  {key}: {metrics[key]:.4f}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Test closed models on medical diagnosis dataset")

    # Model configuration
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["openai", "anthropic"],
                       help="Type of model API to use")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (e.g., gpt-4-vision-preview, claude-3-opus)")
    parser.add_argument("--api-key", type=str, required=True,
                       help="API key for the model")

    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to JSONL dataset file")
    parser.add_argument("--image-base-path", type=str, default=None,
                       help="Base path for images (default: dataset directory)")
    parser.add_argument("--num-examples", type=int, default=None,
                       help="Number of examples to test (default: all)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Directory to save outputs")

    # Processing configuration
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Maximum number of parallel workers")

    # Evaluation
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation after testing")

    args = parser.parse_args()

    # Initialize components
    logger.info("Initializing model and dataset...")

    model_wrapper = ModelAPIWrapper(
        model_type=args.model_type,
        api_key=args.api_key,
        model_name=args.model_name
    )

    dataset_processor = DatasetProcessor(
        dataset_path=args.dataset,
        image_base_path=args.image_base_path
    )

    # Initialize tester
    tester = ModelTester(
        model_wrapper=model_wrapper,
        dataset_processor=dataset_processor,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )

    # Run test
    logger.info("Starting model testing...")
    tester.run_test(num_examples=args.num_examples)

    # Run evaluation
    if not args.skip_evaluation:
        logger.info("Running evaluation...")
        metrics = tester.run_evaluation()

    logger.info("Testing complete!")


if __name__ == "__main__":
    main()