"""
AdaEvolve Controller - Evolution loop with adaptive search intensity.

A clean implementation that uses the adaptive database for all
exploration/exploitation decisions. No explicit stagnation tracking -
search intensity handles exploration automatically.

Features:
- Adaptive sampling based on accumulated improvement signal
- Mode-aware prompting (exploration vs exploitation)
- Paradigm breakthrough for high-level strategy shifts
- Sibling context for learning from previous attempts
- Comprehensive JSON logging of all AdaEvolve signals
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from coding_agent.context_builder.adaevolve import AdaEvolveContextBuilder
from coding_agent.context_builder.default import DefaultContextBuilder
from coding_agent.evaluation.llm_judge import LLMJudge
from coding_agent.llm.llm_pool import LLMPool
from coding_agent.search.adaevolve.paradigm import ParadigmGenerator
from coding_agent.search.base_database import Program
from coding_agent.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from coding_agent.search.utils.discovery_utils import SerializableResult
from coding_agent.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class AdaEvolveController(DiscoveryController):
    """
    AdaEvolve evolution controller with adaptive search intensity.

    Key Features:
    1. Adaptive sampling: Search intensity per island determines exploration/exploitation
    2. Mode-aware prompting: Different guidance for exploration vs exploitation
    3. Sibling context: Shows previous mutations for learning
    4. Error retry: Retries failed generations with error context
    5. Island rotation: UCB-based selection via database.end_iteration()
    6. Paradigm breakthrough: High-level strategy shifts when globally stuck

    No explicit stagnation tracking - search intensity handles exploration
    automatically based on accumulated improvement signal.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Configuration
        db_config = self.config.search.database
        self.enable_retry = getattr(db_config, "enable_error_retry", True)
        self.max_retries = getattr(db_config, "max_error_retries", 2)
        self.num_context_programs = self.config.search.num_context_programs

        # Components
        self.llms = LLMPool(self.config.llm.models)
        self.context_builder = AdaEvolveContextBuilder(self.config)

        # Paradigm generator (if paradigm breakthrough is enabled)
        # Note: We check database.use_paradigm_breakthrough at runtime, not this init-time flag
        # This ensures correct behavior after checkpoint load
        if self.database.use_paradigm_breakthrough:
            model_names = ", ".join(m.name for m in self.guide_llms.models_cfg)
            logger.info(f"Paradigm LLM: using guide_models [{model_names}]")

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_config, "pareto_objectives", []),
                higher_is_better=getattr(db_config, "higher_is_better", {}),
                fitness_key=getattr(db_config, "fitness_key", None),
            )
        else:
            self.paradigm_generator = None

        # JSON logging for comprehensive AdaEvolve stats
        self._iteration_stats_log_path: Optional[str] = None
        self._score_log_path: Optional[str] = None
        self._iteration_stats_file = None
        self._last_sampling_mode: Optional[str] = None
        self._last_sampling_intensity: Optional[float] = None
        self._token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self._reflection_token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self._positive_memory: List[Dict[str, Any]] = []
        self._negative_memory: List[Dict[str, Any]] = []
        self._llm_prompt_log_path: Optional[str] = None

        logger.info(
            f"AdaEvolveController initialized "
            f"(language={self.config.language}, "
            f"paradigm_breakthrough={self.database.use_paradigm_breakthrough})"
        )

    def _load_evaluator_code(self) -> str:
        """Load evaluator source code for paradigm generation context."""
        from coding_agent.search.utils.discovery_utils import load_evaluator_code

        return load_evaluator_code(self.evaluation_file)

    # =========================================================================
    # JSON Logging for AdaEvolve Stats
    # =========================================================================

    def _setup_iteration_stats_logging(self, output_dir: Optional[str] = None) -> None:
        """
        Set up JSON logging for comprehensive iteration statistics.

        Creates a JSONL file that records all AdaEvolve signals at each iteration.
        This enables detailed post-hoc analysis of the discovery process.

        Args:
            output_dir: Directory to write the log file. If None, uses database.config.db_path
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = getattr(self.database.config, "db_path", None)
        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._iteration_stats_log_path = os.path.join(
            output_dir, f"adaevolve_iteration_stats_{timestamp}.jsonl"
        )
        self._score_log_path = os.path.join(output_dir, "convergence.jsonl")
        self._llm_prompt_log_path = (
            os.path.join(output_dir, self.config.llm_prompt_log_file)
            if getattr(self.config, "log_all_llm_prompts", False)
            else None
        )

        logger.info(
            f"AdaEvolve iteration stats will be logged to: {self._iteration_stats_log_path}"
        )
        logger.info(f"AdaEvolve convergence scores will be logged to: {self._score_log_path}")
        if self._llm_prompt_log_path:
            logger.info(f"All LLM prompts will be logged to: {self._llm_prompt_log_path}")

    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
                "token_usage": dict(self._token_usage),
                "reflection_token_usage": dict(self._reflection_token_usage),
            }

            # Add child program info if available
            if child_program:
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                }

            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def get_iteration_stats_log_path(self) -> Optional[str]:
        """Get the path to the iteration stats log file."""
        return self._iteration_stats_log_path

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run evolution with adaptive search intensity and island rotation."""
        total = start_iteration + max_iterations
        logger.info(
            f"AdaEvolve: Running {max_iterations} iterations "
            f"across {self.database.num_islands} islands"
        )

        # Set up comprehensive JSON logging for iteration stats
        self._setup_iteration_stats_logging()

        # Ensure all islands are seeded
        self._ensure_all_islands_seeded()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break
            max_tokens = getattr(self.config.llm, "max_total_tokens", None)
            if max_tokens and self._token_usage["total_tokens"] >= max_tokens:
                logger.info(
                    "Token budget reached: total=%d limit=%d",
                    self._token_usage["total_tokens"],
                    max_tokens,
                )
                break

            try:
                await self._run_iteration(iteration, checkpoint_callback)
            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")
            finally:
                # CRITICAL: Tell database iteration is complete
                # This handles island rotation (UCB) and migration
                self.database.end_iteration(iteration)

        logger.info("AdaEvolve completed")
        self.database.log_status()

        # Log final summary and stats file location
        if self._iteration_stats_log_path:
            logger.info(f"AdaEvolve iteration stats saved to: {self._iteration_stats_log_path}")
        logger.info(
            "Final token usage: total=%d prompt=%d completion=%d requests=%d",
            self._token_usage["total_tokens"],
            self._token_usage["prompt_tokens"],
            self._token_usage["completion_tokens"],
            self._token_usage["requests"],
        )

        return self.database.get_best_program()

    def _ensure_all_islands_seeded(self) -> None:
        """Ensure all islands have at least one program."""
        # Find a seed program
        seed_program = None
        for i in range(self.database.num_islands):
            size = self.database.get_island_size(i)
            if size > 0 and seed_program is None:
                population = self.database.get_island_population(i)
                if population:
                    seed_program = population[0]
                    break

        if seed_program is None:
            logger.warning("No seed program found")
            return

        # Seed empty islands
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) == 0:
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=seed_program.solution,
                    language=seed_program.language,
                    metrics=seed_program.metrics.copy() if seed_program.metrics else {},
                    iteration_found=seed_program.iteration_found,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": i},
                )
                self.database.add(copy, iteration=0, target_island=i)
                logger.info(f"Seeded island {i}")

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        """Execute one evolution iteration."""
        iteration_start_time = time.time()

        # Check for global paradigm stagnation
        # Use database flag directly to stay in sync after checkpoint load
        if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
            await self._generate_paradigms_if_needed()

        result = await self._run_normal_step(iteration)

        iteration_time = time.time() - iteration_start_time

        if result.error:
            logger.warning(f"Iteration {iteration}: {result.error}")
            island_mcts_update = self._record_outer_mcts_failed_iteration(
                iteration=iteration,
                error=result.error,
            )
            self._log_failed_score(
                iteration=iteration,
                result=result,
                island_mcts_update=island_mcts_update,
            )
            # Log failed iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=result.error,
            )
        else:
            self._process_result(result, iteration, checkpoint_callback)
            # Log successful iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=result.child_program_dict,
                iteration_time=result.iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=None,
            )

    def _record_outer_mcts_failed_iteration(
        self,
        *,
        iteration: int,
        error: str,
    ) -> Optional[Dict[str, Any]]:
        """Backpropagate all-attempts-failed iterations to outer island MCTS."""
        record_failure = getattr(self.database, "record_island_mcts_failure", None)
        if not callable(record_failure):
            return None
        try:
            return record_failure(iteration=iteration, error=error)
        except Exception:
            logger.warning("Failed to record outer island-MCTS failure", exc_info=True)
            return None

    async def _generate_paradigms_if_needed(self) -> None:
        """Generate new paradigms if stagnating and none active."""
        if self.paradigm_generator is None:
            return

        if self.database.has_active_paradigm():
            return  # Already have paradigms to use

        logger.info("Global paradigm stagnation detected, generating breakthrough ideas...")

        # Get current best program for context
        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program)

        # Extract evaluator feedback from the best program's artifacts
        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        # Get previously tried ideas for feedback
        previously_tried = self.database.get_previously_tried_ideas()

        # Generate new paradigms
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_solution,
            current_best_score=best_score,
            previously_tried_ideas=previously_tried,
            evaluator_feedback=evaluator_feedback,
        )

        if paradigms:
            self.database.set_paradigms(paradigms)
            logger.info(f"Generated {len(paradigms)} breakthrough paradigms")
        else:
            logger.warning("Failed to generate paradigms")

    async def _run_normal_step(self, iteration: int) -> SerializableResult:
        """Run a normal iteration with optional retry."""
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)

        for attempt in range(attempts):
            result = await self._generate_child(iteration, error_context=last_error)
            if not result.error:
                return result
            last_error = result.error
            logger.debug(f"Attempt {attempt + 1}/{attempts} failed: {last_error}")

        return SerializableResult(
            error=f"All {attempts} attempts failed: {last_error}",
            iteration=iteration,
        )

    def _process_result(
        self,
        result: SerializableResult,
        iteration: int,
        checkpoint_callback,
    ) -> None:
        """Process a successful result by adding to database."""
        child = Program(**result.child_program_dict)

        # Add to database (database handles which island)
        add_result = (
            self.database.add_with_result(child, iteration=iteration, parent_id=result.parent_id)
            if hasattr(self.database, "add_with_result")
            else None
        )
        if add_result is None:
            self.database.add(child, iteration=iteration, parent_id=result.parent_id)

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        # Log prompt
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        # Log progress
        logger.info(
            f"Iteration {iteration}: Program {child.id[:8]} "
            f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}) "
            f"completed in {result.iteration_time:.2f}s"
            f" (llm: {result.llm_generation_time:.2f}s,"
            f" eval: {result.eval_time:.2f}s)"
        )

        # Log metrics
        if child.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in child.metrics.items()
            )
            logger.info(f"Metrics: {metrics_str}")

        # Check for new best
        if self.database.is_multiobjective_enabled():
            pareto_front_ids = {program.id for program in self.database.get_pareto_front()}
            if child.id in pareto_front_ids:
                logger.info(f"Program entered the global Pareto front at iteration {iteration}")
            if self.database.best_program_id == child.id:
                logger.info(f"New representative Pareto solution found at iteration {iteration}")
        elif self.database.best_program_id == child.id:
            logger.info(f"New best solution found at iteration {iteration}")

        self._log_score(iteration, child, add_result, result)

        # Checkpoint callback
        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info(f"Checkpoint interval reached at iteration {iteration}")
            self.database.log_status()
            if checkpoint_callback:
                checkpoint_callback(iteration)

    def _log_score(
        self,
        iteration: int,
        child: Program,
        add_result: Any,
        result: SerializableResult,
    ) -> None:
        """Append plot-friendly per-iteration score data."""
        if not self._score_log_path:
            return
        best = self.database.get_best_program()
        child_score = self.database.get_program_proxy_score(child)
        best_score = self.database.get_program_proxy_score(best)
        record = {
            "iteration": iteration,
            "program_id": child.id,
            "parent_id": child.parent_id,
            "score": child_score if child_score != float("-inf") else None,
            "child_score": child_score if child_score != float("-inf") else None,
            "metrics": child.metrics,
            "best_program_id": best.id if best else None,
            "best_score": best_score if best_score != float("-inf") else None,
            "token_usage": dict(self._token_usage),
            "reflection_token_usage": dict(self._reflection_token_usage),
            "reflection": self._reflection_log_payload(child),
            "mcts": {
                "accepted": getattr(add_result, "accepted", None),
                "reward": getattr(add_result, "reward", None),
                "island_idx": getattr(add_result, "island_idx", None),
                "parent_node_id": getattr(add_result, "parent_node_id", None),
                "child_node_id": getattr(add_result, "child_node_id", None),
                "feature_key": getattr(add_result, "feature_key", None),
            },
            "island_mcts": getattr(add_result, "island_mcts", None),
            "timing": {
                "iteration_time": result.iteration_time,
                "llm_generation_time": result.llm_generation_time,
                "eval_time": result.eval_time,
            },
        }
        with open(self._score_log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _log_failed_score(
        self,
        *,
        iteration: int,
        result: SerializableResult,
        island_mcts_update: Optional[Dict[str, Any]],
    ) -> None:
        """Append a plot-friendly row for all-attempts-failed iterations."""
        if not self._score_log_path:
            return
        best = self.database.get_best_program()
        best_score = self.database.get_program_proxy_score(best)
        record = {
            "iteration": iteration,
            "program_id": None,
            "parent_id": result.parent_id,
            "score": None,
            "child_score": None,
            "metrics": {},
            "best_program_id": best.id if best else None,
            "best_score": best_score if best_score != float("-inf") else None,
            "token_usage": dict(self._token_usage),
            "reflection_token_usage": dict(self._reflection_token_usage),
            "reflection": None,
            "mcts": {
                "accepted": False,
                "reward": None,
                "island_idx": getattr(self.database, "current_island", None),
                "parent_node_id": None,
                "child_node_id": None,
                "feature_key": None,
            },
            "island_mcts": island_mcts_update,
            "error": result.error,
            "timing": {
                "iteration_time": result.iteration_time,
                "llm_generation_time": result.llm_generation_time,
                "eval_time": result.eval_time,
            },
        }
        with open(self._score_log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # =========================================================================
    # Child Generation
    # =========================================================================

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
    ) -> SerializableResult:
        """Generate and evaluate a single child program."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            # Ensure all islands are seeded (needed after from-scratch bootstrap)
            self._ensure_all_islands_seeded()

            # Sample parent and context programs (database returns standard framework dicts)
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
                iteration=iteration,
            )

            # Unpack parent dict (standard framework pattern)
            if not parent_dict:
                logger.error("sample() returned empty parent dict")
                return SerializableResult(
                    error="Empty parent dict from sample()", iteration=iteration
                )
            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]

            # Read sampling mode stashed by database.sample()
            sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"

            # Capture sampling mode and intensity for logging
            self._last_sampling_mode = sampling_mode
            current_island = self.database.current_island
            if self.database.use_adaptive_search:
                self._last_sampling_intensity = self.database.adapter.get_search_intensity(
                    current_island
                )
            else:
                self._last_sampling_intensity = self.database.fixed_intensity

            # When paradigm is active, use best program as parent
            # This ensures paradigm (designed from best) is applied to best, not random parent
            paradigm = (
                self.database.get_current_paradigm()
                if self.database.use_paradigm_breakthrough
                else None
            )
            if paradigm:
                best_program = self.database.get_best_program()
                if best_program:
                    parent_dict = {parent_label: best_program}
                    parent = best_program
                    # Keep context_programs_dict from sampling for diversity

            # Gather siblings for sibling context
            siblings = []
            if hasattr(self.database, "get_children"):
                try:
                    siblings = self.database.get_children(parent.id)
                except (AttributeError, NotImplementedError):
                    pass

            # Build context for prompt generation
            # Only database-derived data — config values are read by the
            # context builder from self.config directly.
            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                # AdaEvolve-specific keys (consumed by AdaEvolveContextBuilder)
                "paradigm": paradigm,
                "siblings": siblings,
                "error_context": error_context,
                "reflection_memory": self._build_reflection_memory(parent),
            }
            search_mode_guidance = getattr(self.database, "_last_search_mode_guidance", None)
            if search_mode_guidance:
                context["search_mode_guidance"] = search_mode_guidance
            # Include any extra prompt context
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            # Build prompt (AdaEvolveContextBuilder handles paradigm/sibling/error formatting)
            prompt = self.context_builder.build_prompt(parent_dict, context)

            # Mark paradigm as used after prompt is built
            if paradigm:
                self.database.use_paradigm()

            # Build tracking info for child program
            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            # Apply human feedback (append or replace mode)
            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            # Generate and evaluate
            return await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    # =========================================================================
    # LLM Generation
    # =========================================================================

    async def _execute_generation(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
    ) -> SerializableResult:
        """Execute LLM generation and evaluation."""
        start_time = time.time()

        image_path = None
        child_id = str(uuid.uuid4())

        # Generate
        llm_generation_time = 0.0
        try:
            llm_start = time.time()
            if self.config.language == "image":
                from coding_agent.search.utils.discovery_utils import build_image_content

                user_content = build_image_content(
                    prompt["user"], parent, other_context_programs or {}
                )
                result = await self._call_llm(
                    prompt["system"],
                    user_content,
                    image_output=True,
                    output_dir=self._get_image_output_dir(),
                    program_id=child_id,
                )
                response = result.text or ""
                self._record_token_usage(result.usage)
                image_path = result.image_path
                if not image_path:
                    return SerializableResult(
                        error="VLM did not generate an image", iteration=iteration
                    )
            else:
                result = await self._call_llm(prompt["system"], prompt["user"])
                response = result.text
                self._record_token_usage(result.usage)
            llm_generation_time = time.time() - llm_start
            self._log_llm_prompt(
                kind="generation",
                iteration=iteration,
                program_id=child_id,
                parent_id=parent.id,
                prompt=prompt,
                response=response,
                usage=result.usage,
                metadata={"language": self.config.language},
            )
        except Exception as e:
            return SerializableResult(error=f"LLM error: {e}", iteration=iteration)

        if not response and self.config.language != "image":
            return SerializableResult(error="Empty LLM response", iteration=iteration)

        # Parse code from response
        if self.config.language == "image":
            child_solution = response or "(image generated)"
            changes = "Image generation"
        elif self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                child_solution = apply_diff(parent.solution, response)
                changes = format_diff_summary(diffs)
            else:
                # No diffs found, try full rewrite
                child_solution = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(error="No valid solution in response", iteration=iteration)

        # Evaluate
        try:
            eval_input = image_path if self.config.language == "image" else child_solution
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
            eval_time = time.time() - eval_start
        except Exception as e:
            return SerializableResult(error=f"Evaluation error: {e}", iteration=iteration)

        metrics = eval_result.metrics
        artifacts = eval_result.artifacts

        # Check for eval failure (e.g., constraint violation, malformed solution)
        if (
            metrics.get("validity") in (0, -1)
            or (metrics.get("timeout") is True and metrics.get("validity") is None)
            or (
                metrics.get("combined_score") == 0
                and (metrics.get("error") is not None or "error" in (artifacts or {}))
            )
        ):
            error_msg = (
                (metrics.get("error") if isinstance(metrics.get("error"), str) else None)
                or (artifacts or {}).get("error")
                or metrics.get("error_message")
                or "Evaluation failed"
            )
            await self._record_failed_reflection_memory(
                parent=parent,
                child_solution=child_solution,
                child_metrics=metrics,
                child_artifacts=artifacts or {},
                changes=changes,
                error_msg=str(error_msg),
                iteration=iteration,
                child_id=child_id,
                prompt=prompt,
            )
            return SerializableResult(error=f"Eval failure: {error_msg}", iteration=iteration)

        # Extract image_path from evaluator metrics (non-image mode fallback)
        if not image_path:
            image_path = (
                metrics.pop("image_path", None)
                if isinstance(metrics.get("image_path"), str)
                else None
            )

        # Build child program with full tracking info
        child_metadata = {"changes": changes, "parent_metrics": parent.metrics}
        if image_path:
            child_metadata["image_path"] = image_path
        reflection = await self._maybe_reflect_parent_child(
            parent=parent,
            child_id=child_id,
            child_solution=child_solution,
            child_metrics=metrics,
            child_artifacts=artifacts or {},
            changes=changes,
            llm_response=response,
            iteration=iteration,
            prompt=prompt,
        )
        if reflection:
            child_metadata["reflection"] = reflection
            if reflection.get("strategy_tags"):
                child_metadata["strategy_tags"] = reflection.get("strategy_tags")
        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            generation=parent.generation + 1,
            metadata=child_metadata,
            artifacts=artifacts,
        )

        iteration_time = time.time() - start_time

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            iteration_time=iteration_time,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            token_usage=result.usage,
            iteration=iteration,
        )

    def _record_token_usage(self, usage: Optional[Dict[str, int]]) -> None:
        if not usage:
            return
        self._token_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._token_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
        if total <= 0:
            total = int(usage.get("prompt_tokens", 0) or 0) + int(
                usage.get("completion_tokens", 0) or 0
            )
        self._token_usage["total_tokens"] += total
        self._token_usage["requests"] += 1

    def _record_reflection_token_usage(self, usage: Optional[Dict[str, int]]) -> None:
        if not usage:
            return
        self._reflection_token_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._reflection_token_usage["completion_tokens"] += int(
            usage.get("completion_tokens", 0) or 0
        )
        total = int(usage.get("total_tokens", 0) or 0)
        if total <= 0:
            total = int(usage.get("prompt_tokens", 0) or 0) + int(
                usage.get("completion_tokens", 0) or 0
            )
        self._reflection_token_usage["total_tokens"] += total
        self._reflection_token_usage["requests"] += 1
        self._record_token_usage(usage)

    def _reflection_enabled(self) -> bool:
        return bool(getattr(self.config.reflection, "enabled", False))

    def _program_score(self, program: Optional[Program]) -> Optional[float]:
        if program is None:
            return None
        score = self.database.get_program_proxy_score(program)
        return None if score == float("-inf") else score

    def _metrics_score(self, metrics: Dict[str, Any]) -> Optional[float]:
        return self._program_score(Program(id="tmp", solution="", metrics=metrics or {}))

    def _truncate(self, text: Any, max_chars: int) -> str:
        value = "" if text is None else str(text)
        if len(value) <= max_chars:
            return value
        return value[: max(0, max_chars - 24)] + "\n... (truncated)"

    def _reflection_log_payload(self, child: Program) -> Dict[str, Any]:
        reflection = (child.metadata or {}).get("reflection") or {}
        if not isinstance(reflection, dict):
            return {"enabled": self._reflection_enabled(), "present": False}
        return {
            "enabled": self._reflection_enabled(),
            "present": bool(reflection),
            "outcome": reflection.get("outcome"),
            "strategy_tags": reflection.get("strategy_tags"),
            "confidence": reflection.get("confidence"),
            "reflection_tokens": reflection.get("token_usage", {}),
        }

    def _build_reflection_memory(self, parent: Program) -> str:
        if not self._reflection_enabled():
            return ""
        cfg = self.config.reflection
        sections: List[str] = []
        reflection = (
            (parent.metadata or {}).get("reflection") if cfg.parent_reflection_in_prompt else None
        )
        if isinstance(reflection, dict) and reflection:
            lines = [
                "## REFLECTION MEMORY FOR CURRENT PROGRAM",
                f"Outcome when created: {reflection.get('outcome', 'unknown')}",
            ]
            summary = reflection.get("mechanism_summary")
            if summary:
                lines.append(f"What this program appears to do: {summary}")
            lesson = reflection.get("why_it_helped") or reflection.get("why_it_failed")
            if lesson:
                lines.append(f"Lesson: {lesson}")
            hints = reflection.get("next_mutation_hints") or []
            if hints:
                lines.append("Promising next mutations:")
                lines.extend(f"- {self._truncate(h, 240)}" for h in hints[:4])
            avoid = reflection.get("avoid") or reflection.get("failure_modes") or []
            if avoid:
                lines.append("Avoid:")
                lines.extend(f"- {self._truncate(a, 240)}" for a in avoid[:4])
            sections.append("\n".join(lines))

        if cfg.global_reflection_in_prompt and self._positive_memory:
            lines = [
                "## RECENT SUCCESS REFLECTION MEMORY",
                "Use these compact lessons from recently evaluated valid programs:",
            ]
            max_items = max(1, int(getattr(cfg, "max_positive_memories", 6)))
            for item in self._positive_memory[-max_items:]:
                summary = item.get("mechanism_summary") or item.get("summary") or ""
                score = item.get("child_score")
                tags = item.get("strategy_tags") or []
                lines.append(
                    f"- score={score}, tags={', '.join(tags[:3])}: "
                    f"{self._truncate(summary, 220)}"
                )
                hints = item.get("next_mutation_hints") or []
                if hints:
                    lines.append(f"  next: {self._truncate(hints[0], 220)}")
            sections.append("\n".join(lines))

        if cfg.negative_memory_in_prompt and self._negative_memory:
            lines = ["## FAILURE MEMORY", "Avoid repeating these recent failure patterns:"]
            for item in self._negative_memory[-cfg.max_negative_memories :]:
                text = item.get("summary") or item.get("error") or ""
                if text:
                    lines.append(f"- {self._truncate(text, 260)}")
            sections.append("\n".join(lines))

        return self._truncate("\n\n".join(sections), cfg.max_prompt_chars) if sections else ""

    def _reflection_pool(self) -> LLMPool:
        role = getattr(self.config.reflection, "model_role", "guide_models")
        if role == "evaluator_models":
            return self.evaluator_llms
        if role == "models":
            return self.llms
        return self.guide_llms

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return {}
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}

    def _reflection_outcome(
        self, parent_score: Optional[float], child_score: Optional[float]
    ) -> str:
        if child_score is None:
            return "unknown"
        if parent_score is None:
            return "new"
        delta = child_score - parent_score
        if delta > 1e-6:
            return "improved"
        if delta < -1e-6:
            return "regressed"
        return "neutral"

    async def _maybe_reflect_parent_child(
        self,
        *,
        parent: Program,
        child_id: str,
        child_solution: str,
        child_metrics: Dict[str, Any],
        child_artifacts: Dict[str, Any],
        changes: str,
        llm_response: str,
        iteration: int,
        prompt: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        if not self._reflection_enabled():
            return None
        cfg = self.config.reflection
        parent_score = self._program_score(parent)
        child_score = self._metrics_score(child_metrics)
        outcome = self._reflection_outcome(parent_score, child_score)
        if outcome == "neutral" and not cfg.reflect_on_neutral:
            return None
        if outcome in {"improved", "regressed", "new"} and not cfg.reflect_on_success:
            return None

        system = (
            "You are a concise scientific-discovery search analyst. "
            "Reflect on one evaluated program mutation. Return only valid JSON."
        )
        max_code_chars = max(1000, cfg.max_context_chars // 3)
        user = f"""
Analyze this parent-child program mutation for future MCTS-guided scientific discovery.

Return JSON with exactly these keys:
- outcome: one of improved, regressed, neutral, new
- strategy_tags: list of 1-3 short snake_case tags
- mechanism_summary: one concise sentence describing the child's approach
- why_it_helped: concise lesson if outcome improved/new, otherwise empty string
- why_it_failed: concise lesson if outcome regressed/neutral, otherwise empty string
- failure_modes: list of 0-2 risks or observed weaknesses
- next_mutation_hints: list of 2-3 actionable future mutation hints
- avoid: list of 0-2 repeated mistakes to avoid
- confidence: number from 0 to 1

Keep every string under 120 characters.

Parent score: {parent_score}
Child score: {child_score}
Observed outcome: {outcome}

Parent metrics:
{json.dumps(parent.metrics or {}, default=str)}

Child metrics:
{json.dumps(child_metrics or {}, default=str)}

Child artifacts:
{json.dumps(child_artifacts or {}, default=str)}

Diff/change summary:
{self._truncate(changes, 1200)}

Parent program:
```{self.config.language or "python"}
{self._truncate(parent.solution, max_code_chars)}
```

Child program:
```{self.config.language or "python"}
{self._truncate(child_solution, max_code_chars)}
```

The generator response that produced the child, if useful:
{self._truncate(llm_response, 2000)}
"""
        try:
            result = await self._reflection_pool().generate(
                system,
                [{"role": "user", "content": user}],
                max_tokens=cfg.max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("Reflection LLM call failed: %s", exc)
            return None

        response_text = result.text or ""
        self._record_reflection_token_usage(result.usage)
        record = self._extract_json_object(response_text)
        if not record:
            record = {
                "outcome": outcome,
                "strategy_tags": [],
                "mechanism_summary": self._truncate(response_text, 600),
                "why_it_helped": "",
                "why_it_failed": "",
                "failure_modes": [],
                "next_mutation_hints": [],
                "avoid": [],
                "confidence": 0.0,
            }
        record.update(
            {
                "version": 1,
                "kind": "parent_child_reflection",
                "created_iteration": iteration,
                "parent_id": parent.id,
                "child_id": child_id,
                "parent_score": parent_score,
                "child_score": child_score,
                "delta": (
                    child_score - parent_score
                    if child_score is not None and parent_score is not None
                    else None
                ),
                "token_usage": result.usage or {},
            }
        )
        self._log_llm_prompt(
            kind="reflection",
            iteration=iteration,
            program_id=child_id,
            parent_id=parent.id,
            prompt={"system": system, "user": user},
            response=response_text,
            usage=result.usage,
            metadata={"outcome": outcome},
            force=cfg.log_reflection_prompts,
        )
        self._append_positive_memory(record)
        return record

    async def _record_failed_reflection_memory(
        self,
        *,
        parent: Program,
        child_solution: str,
        child_metrics: Dict[str, Any],
        child_artifacts: Dict[str, Any],
        changes: str,
        error_msg: str,
        iteration: int,
        child_id: str,
        prompt: Dict[str, str],
    ) -> None:
        if not self._reflection_enabled() or not self.config.reflection.reflect_on_failure:
            self._append_negative_memory(iteration, child_id, error_msg)
            return

        cfg = self.config.reflection
        system = (
            "You are a concise scientific-discovery search analyst. "
            "Summarize a failed generated program as reusable negative search memory. "
            "Return only valid JSON."
        )
        user = f"""
This generated child failed evaluation. Summarize what future MCTS mutations should avoid.

Return JSON with exactly these keys:
- summary: one concise avoid-pattern sentence
- strategy_tags: list of 1-3 short snake_case tags
- avoid: list of 1-2 actionable avoid rules
- repair_hint: one concise hint for fixing this failure mode
- confidence: number from 0 to 1

Keep every string under 120 characters.

Parent score: {self._program_score(parent)}
Failure/error:
{self._truncate(error_msg, 1000)}

Child metrics:
{json.dumps(child_metrics or {}, default=str)}

Child artifacts:
{json.dumps(child_artifacts or {}, default=str)}

Change summary:
{self._truncate(changes, 1000)}

Failed child program:
```{self.config.language or "python"}
{self._truncate(child_solution, cfg.max_context_chars)}
```
"""
        record: Dict[str, Any] = {}
        try:
            result = await self._reflection_pool().generate(
                system,
                [{"role": "user", "content": user}],
                max_tokens=min(cfg.max_tokens, 384),
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            response_text = result.text or ""
            self._record_reflection_token_usage(result.usage)
            record = self._extract_json_object(response_text)
            self._log_llm_prompt(
                kind="failure_reflection",
                iteration=iteration,
                program_id=child_id,
                parent_id=parent.id,
                prompt={"system": system, "user": user},
                response=response_text,
                usage=result.usage,
                metadata={"error": error_msg},
                force=cfg.log_reflection_prompts,
            )
        except Exception as exc:
            logger.warning("Failure reflection LLM call failed: %s", exc)

        summary = record.get("summary") if isinstance(record, dict) else None
        self._append_negative_memory(
            iteration,
            child_id,
            error_msg,
            summary=summary or error_msg,
            reflection=record if isinstance(record, dict) else None,
        )

    def _append_negative_memory(
        self,
        iteration: int,
        child_id: str,
        error: str,
        *,
        summary: Optional[str] = None,
        reflection: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._negative_memory.append(
            {
                "iteration": iteration,
                "program_id": child_id,
                "error": self._truncate(error, 1000),
                "summary": self._truncate(summary or error, 600),
                "reflection": reflection or {},
            }
        )
        max_items = max(1, int(getattr(self.config.reflection, "max_negative_memories", 8)))
        self._negative_memory = self._negative_memory[-max_items:]

    def _append_positive_memory(self, reflection: Dict[str, Any]) -> None:
        if not isinstance(reflection, dict):
            return
        outcome = reflection.get("outcome")
        if outcome not in {"improved", "new", "neutral"}:
            return
        self._positive_memory.append(
            {
                "iteration": reflection.get("created_iteration"),
                "program_id": reflection.get("child_id"),
                "outcome": outcome,
                "child_score": reflection.get("child_score"),
                "strategy_tags": reflection.get("strategy_tags") or [],
                "mechanism_summary": self._truncate(reflection.get("mechanism_summary") or "", 500),
                "next_mutation_hints": reflection.get("next_mutation_hints") or [],
            }
        )
        max_items = max(1, int(getattr(self.config.reflection, "max_positive_memories", 6)))
        self._positive_memory = self._positive_memory[-max_items:]

    def _log_llm_prompt(
        self,
        *,
        kind: str,
        iteration: int,
        program_id: str,
        parent_id: Optional[str],
        prompt: Dict[str, str],
        response: Optional[str],
        usage: Optional[Dict[str, int]],
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        if not self._llm_prompt_log_path and not force:
            return
        if not self._llm_prompt_log_path:
            return
        record = {
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "iteration": iteration,
            "program_id": program_id,
            "parent_id": parent_id,
            "prompt": prompt,
            "response": response,
            "usage": usage or {},
            "metadata": metadata or {},
        }
        try:
            with open(self._llm_prompt_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to log LLM prompt: %s", exc)
