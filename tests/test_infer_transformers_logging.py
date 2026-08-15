from __future__ import annotations

import logging

import eval_tool.infer as infer


def test_processor_kwargs_noise_is_filtered_without_hiding_other_transformers_warnings(
    caplog,
) -> None:
    infer._install_processor_kwargs_warning_filter()
    infer._install_processor_kwargs_warning_filter()
    logger = logging.getLogger("transformers.processing_utils")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.warning(infer._PROCESSOR_KWARGS_WARNING)
        logger.warning("a different transformers warning")

    messages = [record.getMessage() for record in caplog.records]
    assert infer._PROCESSOR_KWARGS_WARNING not in messages
    assert "a different transformers warning" in messages
