import unittest
from unittest.mock import AsyncMock, MagicMock

from agents.executor_agent import ExecutorAgent
from main import AIOPlatform


class SubmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_submission_keeps_current_task_incomplete(self):
        platform = AIOPlatform.__new__(AIOPlatform)
        platform.logger = MagicMock()
        platform.max_iterations = 10
        platform.debug_no_submit = False
        platform.env_client = MagicMock()
        platform.observer = MagicMock()
        platform.observer.analyze_and_decide = AsyncMock(return_value={'submission_command': 'submit()'})
        platform.memory_manager = MagicMock()
        platform.execution_results = {'iterations': [], 'observer_outputs': []}
        # Use the real platform wrapper: it converts environment errors to strings.
        platform.env_client.execute_action.return_value = {'error': True, 'result': 'Repair and verify first'}
        task = platform.observer.get_current_subtask.return_value
        task.is_submit_task = True
        result = await platform.run_iteration(1)
        self.assertNotEqual(result.get('status'), 'completed')
        task.complete_execution.assert_not_called()
        self.assertIn('Repair and verify', platform.observer.previous_iteration_context)

        # Also cover an early completion decision on an ordinary repair task.
        task.is_submit_task = False
        platform.observer.analyze_and_decide.return_value = {
            'next_agent': 'complete', 'current_subtask': task,
            'submission_command': 'submit()',
        }
        result = await platform.run_iteration(2)
        self.assertNotEqual(result.get('status'), 'completed')
        task.complete_execution.assert_not_called()


class ExecutorTest(unittest.TestCase):
    def test_unstructured_context_builds_prompt(self):
        agent = ExecutorAgent.__new__(ExecutorAgent)
        agent.execution_round = 0
        agent.successful_commands = []
        agent.failed_commands = []
        agent.max_iterations = 3
        agent.prompt_loader = MagicMock()
        agent._prepare_input('fix', {}, executor_context='No structured context')
        self.assertEqual(agent.prompt_loader.get_prompt.call_args.kwargs['fix_method'], 'Unknown')

    def test_executor_parses_literal_backslash_in_model_command(self):
        agent = ExecutorAgent.__new__(ExecutorAgent)
        agent.logger = MagicMock()
        result = agent._parse_llm_response(r'''{"executor_command": "exec_shell(\'kubectl get pods\')"}''')
        self.assertEqual(result['executor_command'], "exec_shell('kubectl get pods')")
