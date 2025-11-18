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

load_dotenv()

logger = logging.getLogger(__name__)

# Thread-safe logging lock
log_lock = threading.Lock()

def thread_safe_log(level, message):
    """Thread-safe logging function"""
    with log_lock:
        logger.log(level, message)

def run_single_game(model_name, gamemaster_model, agent_type, game_id, run_id, experiment_dir):
    """
    Run a single game with specified models and agent type
    """
    try:
        # Initialize base agent for 20 Questions
        base_agent = ta.agents.OpenRouterAgent(model_name=model_name)

        # Initialize the 20 Questions environment
        env = ta.make(env_id="TwentyQuestions-v0-raw")
        # Change the gamemaster model (before reset)
        env.gamemaster = ta.agents.OpenRouterAgent(model_name=gamemaster_model)

        # Reset the environment for single player
        env.reset(num_players=1, game_theme="iclr")

        # Extract ground truth after reset
        ground_truth_word = env.game_word
        ground_truth_theme = env.game_theme

        thread_safe_log(logging.INFO, f"🎯 Game {game_id} (Run {run_id}) - Target: '{ground_truth_word}' (theme: {ground_truth_theme})")

        # Initialize agent based on type
        if agent_type == "LLM":
            agent = LLMAgent(base_agent, ground_truth_theme)
        elif agent_type == "Bayes-M":
            agent = BayesMAgent(base_agent, ground_truth_theme)
        elif agent_type == "Bayes-Q":
            agent = BayesQAgent(base_agent, ground_truth_theme, k=10)
        elif agent_type == "Bayes-QM":
            agent = BayesQMAgent(base_agent, ground_truth_theme, k=10)
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
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Save individual game file in games/ subdirectory
        games_dir = os.path.join(experiment_dir, 'games')
        os.makedirs(games_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include microseconds
        filename = os.path.join(games_dir, f'game_{model_name.replace("/", "_")}_{agent_type}_{run_id}_{timestamp}.json')

        with open(filename, 'w') as f:
            json.dump(game_result, f, indent=2)

        thread_safe_log(logging.INFO, f"✅ Game {game_id} (Run {run_id}) completed: {model_name} ({agent_type}) - {turn_count} turns, {game_duration:.1f}s")

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
                           games_per_model=5, max_workers=10, cli_command=None):
    """
    Run multiple games in parallel across different models and agent types

    Args:
        models: List of model names to test
        gamemaster_model: Model to use as gamemaster
        agent_types: List of agent types to test ["LLM", "Bayes-M", "Bayes-Q", "Bayes-QM"]
        games_per_model: Number of games to run per model per agent type
        max_workers: Maximum number of concurrent threads
        cli_command: The original CLI command string for reproducibility
    """
    # Create experiment directory with timestamp
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    experiment_dir = os.path.join('experiments', f'run_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Save CLI command to args.txt
    args_file = os.path.join(experiment_dir, 'args.txt')
    with open(args_file, 'w') as f:
        if cli_command:
            f.write(cli_command + '\n')
        else:
            f.write('# CLI command not available\n')

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
                    'run_id': run
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
    start_time = time.time()

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
                experiment_dir
            ): config for config in game_configs
        }

        # Process completed games
        for future in as_completed(future_to_config):
            result = future.result()
            results.append(result)
            completed_games += 1

            elapsed_time = time.time() - start_time
            avg_time_per_game = elapsed_time / completed_games if completed_games > 0 else 0
            estimated_remaining = (total_games - completed_games) * avg_time_per_game

            thread_safe_log(logging.INFO, f"Progress: {completed_games}/{total_games} games completed "
                            f"({completed_games/total_games*100:.1f}%) - "
                            f"Est. remaining: {estimated_remaining/60:.1f} min")

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
        '--show-prompts',
        action='store_true',
        default=False,
        help='Show LLM prompts at INFO level (default: False)'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress noisy HTTP logs from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure prompt logging
    import agents
    agents.SHOW_PROMPTS = args.show_prompts

    # Construct CLI command for reproducibility
    cli_command = ' '.join(sys.argv)

    # Run experiments with both agent types
    results = run_parallel_experiments(
        models=args.models,
        gamemaster_model=args.gamemaster_model,
        agent_types=args.agent_types,
        games_per_model=args.games_per_model,
        max_workers=args.max_workers,
        cli_command=cli_command
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