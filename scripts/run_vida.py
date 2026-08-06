import torch
import logging
from src.config import ExperimentConfig, load_config
from src.evaluation.runner import CTTARunner


def run_vida(config: ExperimentConfig) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("Starting ViDA experiment")

    runner = CTTARunner(config)
    results = runner.run()

    ds = results.get("dataset", "?")
    logger.info(f"ViDA completed on {ds}")
    logger.info(f"Baseline QWK: {results['baseline_metrics']['qwk']:.4f}")
    logger.info(f"Post-adaptation QWK: {results['final_metrics']['qwk']:.4f}")

    dr = results.get("domain_result", {})
    if dr:
        logger.info(f"Adaptations: {dr.get('num_adaptations', 0)}/{dr.get('total_batches', 0)}")

    if "adaptation_rate" in results:
        logger.info(f"Adaptation rate: {results['adaptation_rate']:.4f}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config("configs/method/vida/idrid.yaml")
    run_vida(config)
