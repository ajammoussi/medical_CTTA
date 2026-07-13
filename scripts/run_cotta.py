import torch
import logging
from src.config import ExperimentConfig, load_config
from src.evaluation.runner import CTTARunner


def run_cotta(config: ExperimentConfig) -> dict:
    """Run CTTA adaptation on target dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting CTTA experiment")

    runner = CTTARunner(config)
    results = runner.run()

    mode = results.get("mode", "single_target")

    if mode == "sequential":
        logger.info(f"CTTA completed.")
        logger.info(f"Source baseline QWK: {results['source_baseline']['qwk']:.4f}")

        # Log per-domain results
        for dr in results.get("domain_results", []):
            logger.info(
                f"  Domain {dr['domain_name']}: "
                f"before={dr['baseline_metrics']['qwk']:.4f} "
                f"after={dr['post_adaptation_metrics']['qwk']:.4f}"
            )

        if "forgetting" in results:
            logger.info(f"Forgetting: {results['forgetting']:.4f}")

        if "adaptation_rate" in results:
            logger.info(f"Adaptation rate: {results['adaptation_rate']:.4f}")
    else:
        logger.info(f"CTTA completed. Final QWK: {results['final_metrics']['qwk']:.4f}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config("configs/default.yaml")
    run_cotta(config)
