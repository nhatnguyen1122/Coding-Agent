import asyncio
import errno
import importlib.util
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import contextmanager
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from coding_agent.config import EvaluatorConfig
from coding_agent.evaluation.evaluation_result import EvaluationResult
from coding_agent.evaluation.llm_judge import LLMJudge
from coding_agent.utils.async_utils import TaskPool
from coding_agent.utils.metrics import format_metrics

logger = logging.getLogger(__name__)
_EVALUATOR_ENV_LOCK = RLock()


def _run_evaluation_subprocess(
    evaluation_file: str,
    program_path: str,
    timeout: int,
    function_name: str = "evaluate",
    env_vars: Optional[Dict[str, str]] = None,
    memory_limit_mb: Optional[int] = None,
) -> Any:
    """Run an evaluator function in a killable subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as result_file:
        result_path = result_file.name

    child_code = r"""
import importlib.util
import os
import pickle
import resource
import sys
import traceback

evaluation_file = sys.argv[1]
program_path = sys.argv[2]
result_path = sys.argv[3]
function_name = sys.argv[4]
memory_limit_mb = int(sys.argv[5]) if sys.argv[5] else 0

if memory_limit_mb > 0:
    limit = memory_limit_mb * 1024 * 1024
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        resource_id = getattr(resource, name, None)
        if resource_id is None:
            continue
        try:
            soft, hard = resource.getrlimit(resource_id)
            new_hard = hard if hard not in (-1, resource.RLIM_INFINITY) else limit
            resource.setrlimit(resource_id, (limit, max(limit, new_hard)))
        except Exception:
            pass

try:
    eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    spec = importlib.util.spec_from_file_location("evaluation_module", evaluation_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {evaluation_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluation_module"] = module
    spec.loader.exec_module(module)
    evaluator_fn = getattr(module, function_name)
    result = evaluator_fn(program_path)
    payload = {"ok": True, "result": result}
except BaseException as exc:
    payload = {
        "ok": False,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "type": type(exc).__name__,
    }

with open(result_path, "wb") as handle:
    pickle.dump(payload, handle)
"""

    child_env = os.environ.copy()
    if env_vars:
        child_env.update(env_vars)

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                evaluation_file,
                program_path,
                result_path,
                function_name,
                str(memory_limit_mb or ""),
            ],
            timeout=timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=child_env,
        )

        try:
            with open(result_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            if proc.returncode and proc.returncode < 0:
                signame = signal.Signals(-proc.returncode).name
                if signame in {"SIGKILL", "SIGSEGV", "SIGBUS", "SIGABRT"}:
                    raise MemoryError(
                        f"Evaluation subprocess terminated by {signame}; "
                        f"possible memory limit exceeded ({memory_limit_mb} MB). "
                        f"stderr: {stderr}"
                    ) from exc
            raise RuntimeError(
                f"Evaluation subprocess failed with returncode {proc.returncode}. "
                f"stderr: {stderr}"
            ) from exc

        if payload.get("ok"):
            return payload.get("result")

        raise RuntimeError(
            f"{payload.get('type', 'EvaluationError')}: {payload.get('error')}\n"
            f"{payload.get('traceback', '')}"
        )
    except subprocess.TimeoutExpired as exc:
        raise asyncio.TimeoutError(f"Evaluation timed out after {timeout}s") from exc
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


class Evaluator:
    """
    Runs the user-provided evaluation function on candidate programs.

    Writes the candidate to a temp file, calls evaluate(program_path), and
    returns an EvaluationResult. Supports optional cascade (multi-stage)
    evaluation and LLM-as-a-judge feedback.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        llm_judge: Optional[LLMJudge] = None,
        max_concurrent: int = 4,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        if not config.evaluation_file:
            raise ValueError("EvaluatorConfig.evaluation_file must be set")

        self.config = config
        self.evaluation_file = config.evaluation_file
        self.program_suffix = config.file_suffix
        self.is_image_mode = config.is_image_mode
        self.llm_judge = llm_judge
        self.task_pool = TaskPool(max_concurrency=max_concurrent)
        self.env_vars = dict(env_vars or {})

        self._load_evaluation_function()
        logger.info(f"Initialized evaluator with {self.evaluation_file}")

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def _load_evaluation_function(self) -> None:
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file not found: {self.evaluation_file}")

        eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        self._module_name = f"_coding_agent_eval_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(self._module_name, self.evaluation_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {self.evaluation_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[self._module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "evaluate"):
            raise AttributeError(f"No evaluate() function in {self.evaluation_file}")

        self.evaluate_function = module.evaluate
        self._eval_module = module
        self._validate_cascade_configuration(module)

    def _validate_cascade_configuration(self, module) -> None:
        if not self.config.cascade_evaluation:
            return
        if not hasattr(module, "evaluate_stage1"):
            logger.warning(
                f"cascade_evaluation is true but {self.evaluation_file} has no evaluate_stage1 — will fall back to direct evaluation"
            )
        elif not hasattr(module, "evaluate_stage2"):
            logger.warning(f"{self.evaluation_file} has evaluate_stage1 but no evaluate_stage2")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
        mode: str = "train",
    ) -> EvaluationResult:
        """Evaluate a program and return scores with optional artifacts.

        Args:
            program_solution: Source code of the candidate program.
            program_id: Optional identifier for logging.
            mode: ``"train"`` or ``"test"``.  Ignored by the Python evaluator
                  (the containerized evaluator passes it to evaluate.sh).
        """
        start_time = time.time()
        label = f" {program_id}" if program_id else ""

        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with tempfile.NamedTemporaryFile(suffix=self.program_suffix, delete=False) as f:
                    f.write(program_solution.encode("utf-8"))
                    temp_path = f.name
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logger.error("Disk full — cannot create temp file")
                    return EvaluationResult(metrics={"error": 0.0, "disk_space_error": True})
                raise

            sidecar_path = None
            if self.is_image_mode:
                sidecar_path = temp_path + ".image_path"
                try:
                    with open(sidecar_path, "w") as sf:
                        sf.write(program_solution)
                except Exception as e:
                    logger.warning(f"Failed to write image sidecar: {e}")

            try:
                if self.config.cascade_evaluation:
                    result = await self._cascade_evaluate(temp_path)
                else:
                    result = await self._run_stage(self.evaluate_function, temp_path)

                eval_result = self._normalize_result(result)

                if self.llm_judge:
                    llm_result = await self.llm_judge.evaluate(program_solution, program_id)
                    if llm_result:
                        for name, value in llm_result.metrics.items():
                            eval_result.metrics[f"llm_{name}"] = value
                        eval_result.artifacts.update(llm_result.artifacts)

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{label} in {elapsed:.2f}s: {format_metrics(eval_result.metrics)}"
                )
                return eval_result

            except asyncio.TimeoutError:
                logger.error(
                    f"Program{label} timed out after {time.time() - start_time:.0f}s (limit: {self.config.timeout}s)"
                )
                return EvaluationResult(metrics={"error": 0.0, "timeout": True})

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed{label}: {e}"
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if sidecar_path and os.path.exists(sidecar_path):
                    os.unlink(sidecar_path)

        logger.error(f"All attempts failed{label}: {last_exception}")
        return EvaluationResult(metrics={"error": 0.0})

    async def evaluate_batch(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple programs concurrently.

        Concurrency is bounded by ``max_concurrent`` (passed at init,
        default 4).

        Args:
            programs: List of ``(solution, program_id)`` tuples.

        Returns:
            List of EvaluationResult in the same order as *programs*.
        """
        return await self.task_pool.gather(
            coros=[self.evaluate_program] * len(programs),
            args_list=list(programs),
        )

    def close(self) -> None:
        """Remove the dynamically loaded evaluation module from sys.modules."""
        sys.modules.pop(getattr(self, "_module_name", None), None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_stage(self, func, program_path: str) -> Any:
        """Run a single evaluation function in a killable subprocess."""
        loop = asyncio.get_running_loop()
        function_name = getattr(func, "__name__", "evaluate")

        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                _run_evaluation_subprocess,
                self.evaluation_file,
                program_path,
                self.config.timeout,
                function_name,
                self.env_vars,
                self.config.memory_limit_mb,
            ),
            timeout=self.config.timeout + 5,
        )

    @contextmanager
    def _scoped_env(self):
        if not self.env_vars:
            yield
            return

        with _EVALUATOR_ENV_LOCK:
            old_values = {k: os.environ.get(k) for k in self.env_vars}
            try:
                os.environ.update(self.env_vars)
                yield
            finally:
                for key, old_value in old_values.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

    def _call_with_env(self, func, program_path: str) -> Any:
        with self._scoped_env():
            return func(program_path)

    def _normalize_result(self, result: Any) -> EvaluationResult:
        if isinstance(result, EvaluationResult):
            return result
        if isinstance(result, dict):
            return EvaluationResult.from_dict(result)

        logger.warning(f"Unexpected result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0})

    async def _cascade_evaluate(self, program_path: str) -> EvaluationResult:
        """Run cascade evaluation: stage1 → threshold check → stage2 → merge."""
        module = self._eval_module

        if not hasattr(module, "evaluate_stage1"):
            return self._normalize_result(
                await self._run_stage(self.evaluate_function, program_path)
            )

        # Stage 1
        try:
            stage1 = self._normalize_result(
                await self._run_stage(module.evaluate_stage1, program_path)
            )
        except asyncio.TimeoutError:
            logger.error(f"Stage 1 timed out ({self.config.timeout}s)")
            return EvaluationResult(
                metrics={"error": 0.0, "timeout": True},
                artifacts={"failure_stage": "stage1"},
            )
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return EvaluationResult(
                metrics={"error": 0.0},
                artifacts={
                    "failure_stage": "stage1",
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        if not self._passes_threshold(stage1.metrics, self.config.cascade_thresholds[0]):
            return stage1

        if not hasattr(module, "evaluate_stage2"):
            return stage1

        # Stage 2
        try:
            stage2 = self._normalize_result(
                await self._run_stage(module.evaluate_stage2, program_path)
            )
        except asyncio.TimeoutError:
            logger.error(f"Stage 2 timed out ({self.config.timeout}s)")
            stage1.metrics["timeout"] = True
            stage1.artifacts["failure_stage"] = "stage2"
            return stage1
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            stage1.artifacts.update({"failure_stage": "stage2", "stage2_stderr": str(e)})
            return stage1

        # Merge stages
        merged_metrics = {
            k: float(v)
            for k, v in {**stage1.metrics, **stage2.metrics}.items()
            if isinstance(v, (int, float)) and k != "error"
        }
        return EvaluationResult(
            metrics=merged_metrics,
            artifacts={**stage1.artifacts, **stage2.artifacts},
        )

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """Check if metrics pass the threshold (combined_score or average)."""
        if not metrics:
            return False

        if "combined_score" in metrics:
            score = metrics["combined_score"]
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        valid = [
            float(v) for k, v in metrics.items() if k != "error" and isinstance(v, (int, float))
        ]
        return (sum(valid) / len(valid)) >= threshold if valid else False
