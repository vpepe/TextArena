import textarena as ta
from dotenv import load_dotenv
import os
from agents import LLMAgent, BayesMAgent, BayesQAgent, BayesQMAgent
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import argparse
import traceback
import logging
import sys
import random
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich.console import Console

load_dotenv()

logger = logging.getLogger(__name__)

# Thread-safe logging lock
log_lock = threading.Lock()

def thread_safe_log(level, message):
    """Thread-safe logging function"""
    with log_lock:
        logger.log(level, message)

def load_word_list(theme_specs=None, category=None):
    """
    Load word list from the TwentyQuestions words file

    Args:
        theme_specs: List of category/theme pairs (e.g., ["basic/places", "hardcore/people"])
        category: Load all themes from this category (e.g., "basic")

    Returns:
        List of words combined from all specified themes
    """
    import importlib.resources
    with importlib.resources.files('textarena.envs.TwentyQuestions').joinpath('twenty_questions_words.json').open('r') as f:
        word_data = json.load(f)

    words = []

    # If category is specified, load all themes from that category
    if category:
        if category not in word_data:
            raise ValueError(f"Category '{category}' not found. Available categories: {list(word_data.keys())}")
        for theme_words in word_data[category].values():
            words.extend(theme_words)

    # If theme_specs is specified, load specific category/theme pairs
    if theme_specs:
        for spec in theme_specs:
            if '/' not in spec:
                raise ValueError(f"Theme spec '{spec}' must be in format 'category/theme'")
            cat, theme = spec.split('/', 1)
            if cat not in word_data:
                raise ValueError(f"Category '{cat}' not found in theme spec '{spec}'")
            if theme not in word_data[cat]:
                raise ValueError(f"Theme '{theme}' not found in category '{cat}'. Available themes: {list(word_data[cat].keys())}")
            words.extend(word_data[cat][theme])

    # Default to test/misc if nothing specified
    if not words:
        words = word_data.get("test", {}).get("misc", [])
        if not words:
            raise ValueError("No words found and default 'test/misc' is not available")

    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)

    return unique_words

def generate_word_assignments(games_per_model, seed, theme_specs=None, category=None):
    """
    Generate deterministic word assignments ensuring same run_id gets same word
    across all agent types and models.

    Args:
        games_per_model: Number of games per model
        seed: Random seed for deterministic shuffling
        theme_specs: List of category/theme pairs (e.g., ["basic/places", "hardcore/people"])
        category: Load all themes from this category (e.g., "basic")

    Returns: dict mapping run_id -> word
    """
    random.seed(seed)
    words = load_word_list(theme_specs=theme_specs, category=category)

    # Shuffle words for variety, but deterministically based on seed
    shuffled_words = words.copy()
    random.shuffle(shuffled_words)

    # Create word assignment: same run_id gets same word across all conditions
    word_assignments = {}
    for run_id in range(1, games_per_model + 1):
        # Use modulo to cycle through words if we need more runs than available words
        word_assignments[run_id] = shuffled_words[(run_id - 1) % len(shuffled_words)]

    return word_assignments

def run_single_game(model_name, gamemaster_model, agent_type, game_id, run_id, experiment_dir, target_word=None, theme_label=None, word_list=None):
    """
    Run a single game with specified models and agent type

    Args:
        target_word: If provided, use this specific word instead of random selection
        theme_label: Label describing the theme(s) for the agent's knowledge
        word_list: List of words to use for agent belief tracking (required when using combined themes)
    """
    try:
        # Initialize base agent for 20 Questions
        base_agent = ta.agents.OpenRouterAgent(model_name=model_name)

        # Initialize the 20 Questions environment
        env = ta.make(env_id="TwentyQuestions-v0-raw")
        # Change the gamemaster model (before reset)
        env.gamemaster = ta.agents.OpenRouterAgent(model_name=gamemaster_model)

        # Reset the environment for single player with assigned word
        # Don't pass game_theme since we're using a specific target_word
        env.reset(num_players=1, target_word=target_word)

        # Extract ground truth after reset
        ground_truth_word = env.game_word
        ground_truth_theme = theme_label if theme_label else "unknown"

        thread_safe_log(logging.DEBUG, f"🎯 Game {game_id} (Run {run_id}) - Target: '{ground_truth_word}' (theme: {ground_truth_theme})")

        # Initialize agent based on type
        if agent_type == "LLM":
            agent = LLMAgent(base_agent, ground_truth_theme, word_list=word_list)
        elif agent_type == "Bayes-M":
            agent = BayesMAgent(base_agent, ground_truth_theme, word_list=word_list)
        elif agent_type == "Bayes-Q":
            agent = BayesQAgent(base_agent, ground_truth_theme, k=10, word_list=word_list)
        elif agent_type == "Bayes-QM":
            agent = BayesQMAgent(base_agent, ground_truth_theme, k=10, word_list=word_list)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        done = False
        turn_count = 0

        all_observations = []
        start_time = time.time()

        while not done:
            # Get current observations
            _, observation = env.get_observation()
            new_observations = [{"player": obs[0], "message": obs[1]} for obs in observation]
            all_observations.extend(new_observations)

            # Get action from agent
            action = agent(all_observations)

            # Step the environment
            done, _ = env.step(action=action)

            turn_count += 1

        # Get final results
        rewards, game_info = env.close()
        _, observation = env.get_observation()
        new_observations = [{"player": obs[0], "message": obs[1]} for obs in observation]
        all_observations.extend(new_observations)

        end_time = time.time()
        game_duration = end_time - start_time

        # Retrieve diagnostics from agent if available
        diagnostics = None
        if hasattr(agent, 'get_diagnostics'):
            diagnostics = agent.get_diagnostics()

        # Create game result
        game_result = {
            'game_id': game_id,
            'run_id': run_id,
            'model_name': model_name,
            'gamemaster_model': gamemaster_model,
            'agent_type': agent_type,
            'observations': all_observations,
            'ground_truth_word': ground_truth_word,
            'ground_truth_theme': ground_truth_theme,
            'rewards': rewards,
            'game_info': game_info,
            'turn_count': turn_count,
            'game_duration': game_duration,
            'timestamp': datetime.datetime.now().isoformat(),
            'diagnostics': diagnostics
        }

        # Save individual game file in games/ subdirectory
        games_dir = os.path.join(experiment_dir, 'games')
        os.makedirs(games_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include microseconds
        filename = os.path.join(games_dir, f'game_{model_name.replace("/", "_")}_{agent_type}_{run_id}_{timestamp}.json')

        with open(filename, 'w') as f:
            json.dump(game_result, f, indent=2)

        thread_safe_log(logging.DEBUG, f"✅ Game {game_id} (Run {run_id}) completed: {model_name} ({agent_type}) - {turn_count} turns, {game_duration:.1f}s")

        return game_result

    except Exception as e:
        tb = traceback.format_exc()
        thread_safe_log(logging.ERROR, f"❌ Game {game_id} (Run {run_id}) failed: {model_name} ({agent_type}) - Error: {str(e)}")
        thread_safe_log(logging.ERROR, f"Traceback:\n{tb}")
        return {
            'game_id': game_id,
            'run_id': run_id,
            'model_name': model_name,
            'gamemaster_model': gamemaster_model,
            'agent_type': agent_type,
            'error': str(e),
            'traceback': tb,
            'timestamp': datetime.datetime.now().isoformat()
        }

def run_parallel_experiments(models, gamemaster_model="openai/gpt-4o", agent_types=["LLM", "Bayes-M", "Bayes-Q", "Bayes-QM"],
                           games_per_model=5, max_workers=10, cli_command=None, seed=42, theme_specs=None, category=None):
    """
    Run multiple games in parallel across different models and agent types

    Args:
        models: List of model names to test
        gamemaster_model: Model to use as gamemaster
        agent_types: List of agent types to test ["LLM", "Bayes-M", "Bayes-Q", "Bayes-QM"]
        games_per_model: Number of games to run per model per agent type
        max_workers: Maximum number of concurrent threads
        cli_command: The original CLI command string for reproducibility
        seed: Random seed for deterministic word selection
        theme_specs: List of category/theme pairs (e.g., ["basic/places", "hardcore/people"])
        category: Load all themes from this category (e.g., "basic")
    """
    # Create experiment directory with timestamp
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    experiment_dir = os.path.join('experiments', f'run_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate deterministic word assignments
    word_assignments = generate_word_assignments(games_per_model, seed=seed, theme_specs=theme_specs, category=category)

    # Load the complete word list for agent belief tracking and shuffle it deterministically
    word_list = load_word_list(theme_specs=theme_specs, category=category)
    random.seed(seed)
    random.shuffle(word_list)

    # Create theme label for display
    if category:
        theme_label = f"category:{category}"
    elif theme_specs:
        theme_label = "+".join(theme_specs)
    else:
        theme_label = "test/misc"

    # Save CLI command to args.txt
    args_file = os.path.join(experiment_dir, 'args.txt')
    with open(args_file, 'w') as f:
        if cli_command:
            f.write(cli_command + '\n')
        else:
            f.write('# CLI command not available\n')

    # Save word assignments to file
    word_assignments_file = os.path.join(experiment_dir, 'word_assignments.json')
    with open(word_assignments_file, 'w') as f:
        json.dump(word_assignments, f, indent=2)

    # Set up file logging in addition to console
    log_file = os.path.join(experiment_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)

    thread_safe_log(logging.INFO, f"📁 Experiment directory: {experiment_dir}")
    thread_safe_log(logging.INFO, f"📝 Log file: {log_file}")

    # Create all game configurations
    game_configs = []
    game_id = 1

    for model in models:
        for agent_type in agent_types:
            for run in range(1, games_per_model + 1):
                game_configs.append({
                    'model_name': model,
                    'gamemaster_model': gamemaster_model,
                    'agent_type': agent_type,
                    'game_id': game_id,
                    'run_id': run,
                    'target_word': word_assignments[run],  # Assign word based on run_id
                    'theme_label': theme_label,
                    'word_list': word_list
                })
                game_id += 1

    total_games = len(game_configs)
    thread_safe_log(logging.INFO, f"🚀 Starting {total_games} games with {max_workers} workers...")
    thread_safe_log(logging.INFO, f"Gamemaster: {gamemaster_model}")
    thread_safe_log(logging.INFO, f"Models: {models}")
    thread_safe_log(logging.INFO, f"Agent types: {agent_types}")
    thread_safe_log(logging.INFO, f"Games per model per agent: {games_per_model}")

    # Run games in parallel
    results = []
    completed_games = 0
    failed_games = 0
    start_time = time.time()

    # Create progress bar with Rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Running games...", total=total_games)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all games
            future_to_config = {
                executor.submit(
                    run_single_game,
                    config['model_name'],
                    config['gamemaster_model'],
                    config['agent_type'],
                    config['game_id'],
                    config['run_id'],
                    experiment_dir,
                    config['target_word'],
                    config['theme_label'],
                    config['word_list']
                ): config for config in game_configs
            }

            # Process completed games
            for future in as_completed(future_to_config):
                result = future.result()
                results.append(result)
                completed_games += 1

                # Track failures
                if 'error' in result:
                    failed_games += 1

                # Update progress bar
                progress.update(task, advance=1)

    # Save summary results
    summary = {
        'experiment_summary': {
            'total_games': total_games,
            'completed_games': len([r for r in results if 'error' not in r]),
            'failed_games': len([r for r in results if 'error' in r]),
            'models_tested': models,
            'agent_types_tested': agent_types,
            'games_per_model_per_agent': games_per_model,
            'total_duration': time.time() - start_time,
            'timestamp': datetime.datetime.now().isoformat()
        },
        'results': results
    }

    summary_filename = os.path.join(experiment_dir, 'summary.json')

    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    thread_safe_log(logging.INFO, f"\n🎉 Experiment completed!")
    thread_safe_log(logging.INFO, f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    thread_safe_log(logging.INFO, f"Results saved to: {experiment_dir}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 20 Questions experiments with different models and agents')

    parser.add_argument(
        '--models',
        nargs='+',
        default=["openai/gpt-4o"],
        help='List of models to test (space-separated). Example: --models openai/gpt-4o meta-llama/llama-3.1-8b-instruct'
    )

    parser.add_argument(
        '--gamemaster-model',
        type=str,
        default="openai/gpt-4o",
        help='Model to use as gamemaster (default: openai/gpt-4o)'
    )

    parser.add_argument(
        '--games-per-model',
        type=int,
        default=5,
        help='Number of games to run per model per agent type (default: 5)'
    )

    parser.add_argument(
        '--agent-types',
        nargs='+',
        choices=['LLM', 'Bayes-M', 'Bayes-Q', 'Bayes-QM'],
        default=['LLM', 'Bayes-QM'],
        help='Agent types to test (default: LLM Bayes-QM)'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Maximum number of concurrent threads (default: 10)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        "--show-prompts",
        nargs="*",
        choices=[
            "DECISION",
            "QUESTION",
            "MOVE",
            "CONSISTENCY",
            "ALL",
        ],
        default=None,
        help="Show specific LLM prompt types at INFO level. Options: DECISION, QUESTION, MOVE, CONSISTENCY, ALL. Use --show-prompts without arguments to show all prompts. Example: --show-prompts QUESTION MOVE",
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic word selection (default: 42)'
    )

    parser.add_argument(
        '--themes',
        nargs='+',
        default=None,
        help='List of category/theme pairs to use (e.g., basic/places hardcore/people). See twenty_questions_words.json for available themes.'
    )

    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Load all themes from this category (e.g., basic, hardcore, test). Mutually exclusive with --themes.'
    )

    args = parser.parse_args()

    # Configure logging with Rich
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=False)]
    )

    # Suppress noisy HTTP logs from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure prompt logging
    import agents
    if args.show_prompts is not None:
        # If --show-prompts is used without arguments, show all prompts
        if len(args.show_prompts) == 0 or 'ALL' in args.show_prompts:
            agents.ENABLED_PROMPT_TYPES = {'DECISION', 'QUESTION', 'MOVE', 'CONSISTENCY', 'BATCH_QUESTION'}
        else:
            # Normalize BATCH_QUESTION to "BATCH QUESTION" (with space) as used in agents.py
            enabled = set(args.show_prompts)
            if 'BATCH_QUESTION' in enabled:
                enabled.remove('BATCH_QUESTION')
                enabled.add('BATCH QUESTION')
            agents.ENABLED_PROMPT_TYPES = enabled
    else:
        agents.ENABLED_PROMPT_TYPES = set()

    # Construct CLI command for reproducibility
    cli_command = ' '.join(sys.argv)

    # Validate theme arguments
    if args.themes and args.category:
        parser.error("--themes and --category are mutually exclusive. Use only one.")

    # Run experiments with both agent types
    results = run_parallel_experiments(
        models=args.models,
        gamemaster_model=args.gamemaster_model,
        agent_types=args.agent_types,
        games_per_model=args.games_per_model,
        max_workers=args.max_workers,
        cli_command=cli_command,
        seed=args.seed,
        theme_specs=args.themes,
        category=args.category
    )

    # Print some summary statistics
    successful_games = [r for r in results if 'error' not in r]
    failed_games = [r for r in results if 'error' in r]

    logger.info(f"\n📊 Final Summary:")
    logger.info(f"Successful games: {len(successful_games)}")
    logger.info(f"Failed games: {len(failed_games)}")

    if successful_games:
        avg_turns = sum(r['turn_count'] for r in successful_games) / len(successful_games)
        avg_duration = sum(r['game_duration'] for r in successful_games) / len(successful_games)
        logger.info(f"Average turns per game: {avg_turns:.1f}")
        logger.info(f"Average game duration: {avg_duration:.1f}s")
