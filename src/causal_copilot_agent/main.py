#!/usr/bin/env python
import sys
import warnings
import argparse
from datetime import datetime

from causal_copilot_agent.crew import CausalCopilotAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run(inputs):
    """
    Run the crew.
    """
    try:
        CausalCopilotAgent().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def train(inputs, n_iterations, filename):
    """
    Train the crew for a given number of iterations.
    """
    try:
        CausalCopilotAgent().crew().train(n_iterations=n_iterations, filename=filename, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay(task_id):
    """
    Replay the crew execution from a specific task.
    """
    try:
        CausalCopilotAgent().crew().replay(task_id=task_id)
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test(inputs, n_iterations, eval_llm):
    """
    Test the crew execution and return the results.
    """
    try:
        CausalCopilotAgent().crew().test(n_iterations=n_iterations, eval_llm=eval_llm, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Run the CausalCopilotAgent crew.")
    parser.add_argument("action", choices=["run", "train", "replay", "test"], help="Action to perform")
    parser.add_argument("--inputs", type=str, default="AI LLMs", help="Topic or inputs for the crew")
    parser.add_argument("--current_year", type=str, default=str(datetime.now().year), help="Current year")
    parser.add_argument("--n_iterations", type=int, help="Number of iterations for training or testing")
    parser.add_argument("--filename", type=str, help="Filename for training output")
    parser.add_argument("--task_id", type=str, help="Task ID to replay from")
    parser.add_argument("--eval_llm", type=str, help="Evaluation LLM for testing")

    args = parser.parse_args()

    # Prepare inputs
    inputs = {
        'causal_results': 'The analysis of the causal relationships among the given variables indicates a structured interplay between the signaling pathways. Specifically, P38 and Jnk are identified as direct causal influences on PKC, suggesting that these kinases regulate PKC’s activity, likely playing a role in crucial cellular processes like differentiation and stress response. On the other hand, the undirected relationships among Raf, Mek, Plcg, PIP2, and PIP3 suggest a more complex interaction without clear cause-and-effect, highlighting a network of signaling components that may function collaboratively. For instance, Raf and Mek are known to be part of the MAPK pathway, where Raf activates Mek, thus indicating a pivotal role in cell proliferation and survival. Similarly, Plcg acts on PIP3, influencing downstream signaling involving PIP2, which might be integral for the activation of pathways like Akt that lead to cellular growth. Furthermore, Erk interacts with PKA and Akt, both of which are crucial for various cellular functions including metabolism and cell cycle regulation. Therefore, while P38 and Jnk are clearly defined in their causal roles with PKC, the undirected relationships illustrate a nuanced signaling landscape where feedback and network cooperation are significant. P38 and Jnk cause PKC, indicating their roles in modulating cellular responses. Raf and Mek have a collaborative relationship, functioning within the MAPK pathway for cellular proliferation. Plcg interacts with PIP3, which is crucial for downstream signaling, potentially affecting PIP2 levels. Erks interaction with PKA and Akt demonstrates the intersection of multiple signaling pathways crucial for cell function and metabolism. The overall structure reflects a balance of directed and undirected relationships crucial for proper cellular signaling dynamics.'
        }

    # Perform the requested action
    if args.action == "run":
        run(inputs)
    elif args.action == "train":
        if args.n_iterations is None or args.filename is None:
            raise ValueError("Both --n_iterations and --filename are required for training.")
        train(inputs, args.n_iterations, args.filename)
    elif args.action == "replay":
        if args.task_id is None:
            raise ValueError("--task_id is required for replay.")
        replay(args.task_id)
    elif args.action == "test":
        if args.n_iterations is None or args.eval_llm is None:
            raise ValueError("Both --n_iterations and --eval_llm are required for testing.")
        test(inputs, args.n_iterations, args.eval_llm)

if __name__ == "__main__":
    main()
