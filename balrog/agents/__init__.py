class AgentFactory:
    """Factory class for creating agents based on configuration.

    The `AgentFactory` class is responsible for initializing the appropriate agent type
    based on the provided configuration, which includes setting up the LLM client and
    prompt builder.
    """

    def __init__(self, config):
        """Initialize the AgentFactory with configuration settings.

        Args:
            config (omegaconf.DictConfig): Configuration object containing settings for the agent and client.
        """
        self.config = config

    def create_agent(self, budget_ledger=None):
        """Create an agent instance based on the agent type specified in the configuration.

        The function uses the `config.agent.type` attribute to determine which agent to create.
        It supports several agent types, including Naive, Chain-of-Thought, Self-Refine, Dummy,
        and Custom agents.

        Returns:
            Agent: An instance of the selected agent type, configured with the client and prompt builder.

        Raises:
            ValueError: If an unknown agent type is specified in the configuration.
        """
        # Delay client imports until an evaluation actually constructs an agent.
        # This keeps parser and configuration tests offline and does not change
        # the client selected for a real BALROG run.
        from balrog.client import create_llm_client
        from balrog.prompt_builder import create_prompt_builder

        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        if self.config.agent.type == "naive":
            if budget_ledger is not None:
                from assignment2.agents import LoggedNaiveAgent

                return LoggedNaiveAgent(client_factory, prompt_builder, config=self.config, budget_ledger=budget_ledger)
            from .naive import NaiveAgent

            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "cot":
            from .chain_of_thought import ChainOfThoughtAgent

            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "dummy":
            from .dummy import DummyAgent

            return DummyAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "custom":
            from .custom import CustomAgent

            return CustomAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "few_shot":
            from .few_shot import FewShotAgent

            return FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history)
        elif self.config.agent.type == "robust_naive":
            from .robust_naive import RobustNaiveAgent

            return RobustNaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "robust_cot":
            from .robust_cot import RobustCoTAgent

            return RobustCoTAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "planning":
            from assignment2.agents import PlanningAgent

            return PlanningAgent(client_factory, prompt_builder, config=self.config, budget_ledger=budget_ledger)
        elif self.config.agent.type == "reflective_planning":
            from assignment2.agents import ReflectivePlanningAgent

            return ReflectivePlanningAgent(
                client_factory, prompt_builder, config=self.config, budget_ledger=budget_ledger
            )

        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")
