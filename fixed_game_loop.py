import textarena as ta
from dotenv import load_dotenv
import os
from agents import LLMAgent, EIGAgent
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

load_dotenv()

# Thread-safe print lock
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def run_single_game(model_name, gamemaster_model, agent_type, game_id, run_id):
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
        env.reset(num_players=1)

        # Extract ground truth after reset
        ground_truth_word = env.game_word
        ground_truth_theme = env.game_theme

        # Initialize agent based on type
        if agent_type == "LLM":
            agent = LLMAgent(base_agent, ground_truth_theme)
        else:
            agent = EIGAgent(base_agent, ground_truth_theme)

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

        # Save individual game file
        os.makedirs('experiments', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include microseconds
        filename = f'experiments/game_{model_name.replace("/", "_")}_{agent_type}_{run_id}_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(game_result, f, indent=2)

        thread_safe_print(f"✅ Game {game_id} (Run {run_id}) completed: {model_name} ({agent_type}) - {turn_count} turns, {game_duration:.1f}s - {filename}")

        return game_result

    except Exception as e:
        thread_safe_print(f"❌ Game {game_id} (Run {run_id}) failed: {model_name} ({agent_type}) - Error: {str(e)}")
        return {
            'game_id': game_id,
            'run_id': run_id,
            'model_name': model_name,
            'gamemaster_model': gamemaster_model,
            'agent_type': agent_type,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }

def run_parallel_experiments(models, gamemaster_model="openai/gpt-4o", agent_types=["LLM", "EIG"],
                           games_per_model=5, max_workers=10):
    """
    Run multiple games in parallel across different models and agent types

    Args:
        models: List of model names to test
        gamemaster_model: Model to use as gamemaster
        agent_types: List of agent types to test ["LLM", "EIG"]
        games_per_model: Number of games to run per model per agent type
        max_workers: Maximum number of concurrent threads
    """
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
    thread_safe_print(f"🚀 Starting {total_games} games with {max_workers} workers...")
    thread_safe_print(f"Models: {models}")
    thread_safe_print(f"Agent types: {agent_types}")
    thread_safe_print(f"Games per model per agent: {games_per_model}")

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
                config['run_id']
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

            thread_safe_print(f"Progress: {completed_games}/{total_games} games completed "
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

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f'experiments/experiment_summary_{timestamp}.json'

    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    thread_safe_print(f"\n🎉 Experiment completed!")
    thread_safe_print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    thread_safe_print(f"Summary saved to: {summary_filename}")

    return results

if __name__ == "__main__":
    # Example: Run parallel experiments with multiple models
    models_to_test = [
        "meta-llama/llama-4-scout",
        "openai/gpt-4o",
    ]

    # Run experiments with both agent types
    results = run_parallel_experiments(
        models=models_to_test,
        gamemaster_model="openai/gpt-5",
        agent_types=["LLM", "EIG"],
        games_per_model=120,  # 3 games per model per agent type
        max_workers=128      # 8 concurrent threads
    )

    # Print some summary statistics
    successful_games = [r for r in results if 'error' not in r]
    failed_games = [r for r in results if 'error' in r]

    print(f"\n📊 Final Summary:")
    print(f"Successful games: {len(successful_games)}")
    print(f"Failed games: {len(failed_games)}")

    if successful_games:
        avg_turns = sum(r['turn_count'] for r in successful_games) / len(successful_games)
        avg_duration = sum(r['game_duration'] for r in successful_games) / len(successful_games)
        print(f"Average turns per game: {avg_turns:.1f}")
        print(f"Average game duration: {avg_duration:.1f}s")