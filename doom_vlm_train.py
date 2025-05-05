import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import time
import random
import wandb
import os
from collections import deque

# Imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import vizdoom

# Custom Doom Env
class VizDoomGym(Env):
    def __init__(self, render=False, config='ViZDoom/scenarios/health_gathering.cfg'):
        super().__init__()
        self.game = vizdoom.DoomGame()
        self.game.load_config(config)
        
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
            
        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.action_space = Discrete(3)
    
    def step(self, action, tics):
        actions = np.identity(3)
        movement_reward = self.game.make_action(actions[action], tics)
        
        reward = 0
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            game_variables = self.game.get_state().game_variables
            health = game_variables
            reward = movement_reward
            info = health
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0
            
        info = {"info": info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info
    
    def render(self):
        pass
    
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return state
    
    def close(self):
        self.game.close()

# Value Head for the VLM model
class VLMWithValueHead(nn.Module):
    def __init__(self, vlm_model, device="cuda"):
        super(VLMWithValueHead, self).__init__()
        self.vlm_model = vlm_model
        self.value_head = nn.Sequential(
            nn.Linear(vlm_model.config.hidden_size, 256, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(256, 1, dtype=torch.bfloat16)
        ).to(device)
        self.device = device
    
    def forward(self, inputs):
        outputs = self.vlm_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        value = self.value_head(hidden_states)
        return outputs, value

# PPO Agent
class PPOAgent:
    def __init__(self, model, processor, device="cuda", lr=3e-5, clip_ratio=0.2, gamma=0.99, lambda_gae=0.95):
        self.model = model
        self.processor = processor
        self.device = device
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def get_action_and_value(self, states, infos, device = 'cuda'):
    
        if isinstance(states, np.ndarray):
            states = [states]
        if isinstance(infos, dict):
            infos = [infos]

        messages_batch = []
        images_batch = []

        for state, info in zip(states, infos):
        
            if state.ndim == 3 and state.shape[0] in [1, 3]:
                state = np.transpose(state, (1, 2, 0))
            elif state.ndim == 2:
                state = np.stack([state] * 3, axis=-1)
            img = Image.fromarray(state.astype(np.uint8))
            health = str(info["info"])

            messages_batch.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {
                            "type": "text",
                            "text": f"""You are in a game environment where your life is constantly decreasing, collect health 
                            packages (white gray) present in the environment to survive (image). You are given with the current state of the game, 
                            you need to choose and action: MOVELEFT, MOVERIGHT, STRAIGHT. To collect the health boxes move over them.

                            Write out the reason why you are choosing an action in one concise sentence and choose an action in the shown format.
                            Current health: {health}.

                            Output format: {{ REASON: (reasoning in one line), ACTION: (one of MOVELEFT, MOVERIGHT, STRAIGHT) }}"""
                        }
                    ]
                }
            ])
            images_batch.append(img)
    
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)
    
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs, value= self.model(inputs)

    
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        action_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)

    
        generated_ids = self.model.vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        actions = []
        log_probs = []

        for i, text in enumerate(output_texts):
            text_action_part = text.split(":")[-1].strip().upper()
            if "MOVERIGHT" in text_action_part:
                action = 1
            elif "MOVELEFT" in text_action_part:
                action = 0
            elif "STRAIGHT" in text_action_part:
                action = 2
            else:
                action = 2 

            actions.append(action)
            log_probs.append(torch.log(probs[i, action_token_ids[i]]).item())

    
        out = [
            (actions[i], value[i].item(), log_probs[i], output_texts[i])
            for i in range(len(states))
        ]
        return out[0][0], out[0][1], out[0][2], out[0][3]

    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[-1]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages, infos):
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        messages = []
        all_log_probs = []
        all_values = []

        i = 0
        for state, info in zip(states, infos):

            state = state.transpose(1, 2, 0) 
            img = Image.fromarray(state)
            health = str(info['info']) 
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"""You are in a game environment where your life is constantly decreasing, collect health 
                    boxes (white gray) present in the environment to survive (image). You are given with the current state of the game, 
                    you need to choose and action: MOVELEFT, MOVERIGHT, STRAIGHT. To collect the health boxes move over them.
                    
                    Write out the reason why you are choosing an action in one concise sentence and choose an action in the shown format.
                    Current health: {health}.
                    
                    Output format: {{ REASON: (reasoning in one line), ACTION: (one of MOVELEFT, MOVERIGHT, STRAIGHT) }}"""}
                ]
            })
        
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            outputs, values = self.model(inputs)
            logits = outputs.logits[:, -1, :]  
            probs = torch.nn.functional.softmax(logits, dim=-1)

            action = actions[i].item()
            i += 1
            log_prob = torch.log(probs[0, action])
            
            all_log_probs.append(log_prob)
            all_values.append(values.squeeze())
        
        log_probs = torch.stack(all_log_probs)
        values = torch.stack(all_values)
        
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_fraction = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
        
        return {
            'policy_loss': policy_loss.item(), 
            'value_loss': value_loss.item(), 
            'total_loss': loss.item(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }


# PPO Trainer
class PPOTrainer:
    def __init__(self, agent, env, batch_size=64, epochs=2, tics=4):
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.epochs = epochs
        self.tics = tics
        
        self.episode_counter = 0
        self.global_step = 0
        
    def collect_trajectories(self, num_timesteps):
        states, actions, rewards, values, log_probs, dones, infos = [], [], [], [], [], [], []
        observations = []
        reasonings = []
        
        obs = self.env.reset()

        done = False
        episode_reward = 0
        info = {'info': np.array([100.0])}
        health_info = info['info'][0].item()
        
        episode_rewards = []
        current_episode_reward = 0
        current_episode_length = 0
        action_counts = {0: 0, 1: 0, 2: 0}
        
        for step in range(num_timesteps):
            self.global_step += 1
            
            action, value, log_prob, reasoning = self.agent.get_action_and_value(obs, info)
            
            if step % 10 == 0:
                try:
                    wandb.log({"reasoning": reasoning}, step=self.global_step)
                except Exception as e:
                    print(f"Warning: Failed to log reasoning to wandb: {e}")
                
            action_counts[action] += 1
            
            next_obs, reward, done, info = self.env.step(action, self.tics)
            
            current_health_info = info['info'][0].item() if isinstance(info['info'], np.ndarray) else info['info']
            if health_info >= current_health_info:
                health_info = current_health_info
                reward = reward - self.tics
            else:
                reward = 10 * self.tics
                health_info = current_health_info
            
            states.append(obs)
            observations.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            reasonings.append(reasoning)
            infos.append(info)
            
            current_episode_reward += reward
            current_episode_length += 1
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                self.episode_counter += 1
                print(f"Episode {self.episode_counter} finished with reward {episode_reward}")
                
                try:
                    wandb.log({
                        "episode_reward": current_episode_reward,
                        "episode_length": current_episode_length,
                        "health_at_end": health_info,
                        "episode": self.episode_counter
                    }, step=self.global_step)
                except Exception as e:
                    print(f"Warning: Failed to log episode data to wandb: {e}")
                

                episode_rewards.append(current_episode_reward)
                

                obs = self.env.reset()
                episode_reward = 0
                info = {'info': np.array([100.0])}
                health_info = info['info'][0].item()
                current_episode_reward = 0
                current_episode_length = 0
            
        if episode_rewards:
            try:
                total_actions = sum(action_counts.values())
                action_percents = {
                    "action_moveLeft_percent": (action_counts[0] / total_actions * 100) if total_actions > 0 else 0,
                    "action_moveRight_percent": (action_counts[1] / total_actions * 100) if total_actions > 0 else 0,
                    "action_straight_percent": (action_counts[2] / total_actions * 100) if total_actions > 0 else 0,
                }
                
                wandb.log({
                    "mean_episode_reward": np.mean(episode_rewards),
                    "min_episode_reward": np.min(episode_rewards),
                    "max_episode_reward": np.max(episode_rewards),
                    **action_percents
                }, step=self.global_step)
            except Exception as e:
                print(f"Warning: Failed to log aggregated metrics to wandb: {e}")
                
        return states, observations, actions, rewards, values, log_probs, dones, reasonings, infos
    
    def train(self, num_timesteps=300, num_iterations=4, save_models=False, save_dir=None):
        overall_training_metrics = {
            "policy_losses": [],
            "value_losses": [],
            "total_losses": [],
            "clip_fractions": [],
            "kl_divergences": [],
        }
        
        for iteration in range(num_iterations):
            iteration_start_time = time.time()
            print(f"Iteration {iteration+1}/{num_iterations}")
            
            states, observations, actions, rewards, values, log_probs, dones, reasonings, infos = self.collect_trajectories(num_timesteps)
            
            advantages, returns = self.agent.compute_advantages(rewards, values, dones)
            
            iteration_metrics = {
                "policy_losses": [],
                "value_losses": [],
                "total_losses": [],
                "clip_fractions": [],
                "kl_divergences": [],
            }
            
            # Update the policy network
            for epoch in range(self.epochs):
                indices = np.random.permutation(len(states))
                for start_idx in range(0, len(states), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(states))
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_states = [states[i] for i in batch_indices]
                    batch_actions = [actions[i] for i in batch_indices]
                    batch_log_probs = [log_probs[i] for i in batch_indices]
                    batch_returns = [returns[i] for i in batch_indices]
                    batch_advantages = [advantages[i] for i in batch_indices]
                    batch_infos = [infos[i] for i in batch_indices]
                    
                    loss_info = self.agent.update(
                        batch_states, 
                        batch_actions, 
                        batch_log_probs, 
                        batch_returns, 
                        batch_advantages,
                        batch_infos
                    )
                    
                    iteration_metrics["policy_losses"].append(loss_info["policy_loss"])
                    iteration_metrics["value_losses"].append(loss_info["value_loss"])
                    iteration_metrics["total_losses"].append(loss_info["total_loss"])
                    iteration_metrics["clip_fractions"].append(loss_info["clip_fraction"])
                    iteration_metrics["kl_divergences"].append(loss_info["approx_kl"])
                    
                    print(f"Loss: {loss_info}")
            
            avg_metrics = {
                "policy_loss": np.mean(iteration_metrics["policy_losses"]),
                "value_loss": np.mean(iteration_metrics["value_losses"]),
                "total_loss": np.mean(iteration_metrics["total_losses"]),
                "clip_fraction": np.mean(iteration_metrics["clip_fractions"]),
                "approx_kl": np.mean(iteration_metrics["kl_divergences"]),
                "iteration": iteration + 1,
                "iteration_time": time.time() - iteration_start_time
            }
            
            try:
                wandb.log(avg_metrics, step=self.global_step)
            except Exception as e:
                print(f"Warning: Failed to log iteration metrics to wandb: {e}")
            
            for key in overall_training_metrics:
                overall_training_metrics[key].extend(iteration_metrics[key])
            
            if save_models and save_dir and (iteration + 1) % 20 == 0:
                os.makedirs(save_dir, exist_ok=True)
                model_path = f"{save_dir}/vlm_ppo_model_iter_{iteration+1}.pt"
                torch.save(self.agent.model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
                
        if save_models and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            final_model_path = f"{save_dir}/vlm_ppo_model_final.pt"
            torch.save(self.agent.model.state_dict(), final_model_path)
            print(f"Final model saved to {final_model_path}")
        
        try:
            wandb.run.summary.update({
                "final_policy_loss": np.mean(overall_training_metrics["policy_losses"][-100:]) if overall_training_metrics["policy_losses"] else 0,
                "final_value_loss": np.mean(overall_training_metrics["value_losses"][-100:]) if overall_training_metrics["value_losses"] else 0,
                "final_total_loss": np.mean(overall_training_metrics["total_losses"][-100:]) if overall_training_metrics["total_losses"] else 0,
                "final_clip_fraction": np.mean(overall_training_metrics["clip_fractions"][-100:]) if overall_training_metrics["clip_fractions"] else 0,
                "final_kl_divergence": np.mean(overall_training_metrics["kl_divergences"][-100:]) if overall_training_metrics["kl_divergences"] else 0,
                "total_episodes": self.episode_counter,
                "total_training_steps": self.global_step
            })
        except Exception as e:
            print(f"Warning: Failed to update wandb summary: {e}")

# Main execution
def main():
    run = wandb.init(
        project="vlm-doom-ppo",
        config={
            "model": "Qwen2.5-VL-3B-Instruct",
            "environment": "VizDoom-HealthGathering",
            "batch_size": 4,
            "epochs": 2,
            "tics": 4,
            "lr": 3e-4,
            "clip_ratio": 0.2,
            "gamma": 0.99,
            "lambda_gae": 0.95,
            "num_timesteps": 200,
            "num_iterations": 40,
            "save_models": True,  # New config parameter
            "save_dir": "/projects/p32722/Models/vlm_trained_doom/"  # Default save directory
        }
    )
    
    # Initialize device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    

    wandb.config.update({"device": device})
    
    # Load VLM model and processor
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype="auto", 
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Create model with value head
    model = VLMWithValueHead(vlm_model, device=device)
    

    wandb.config.update({
        "model_hidden_size": vlm_model.config.hidden_size,
        "value_head_dim": 256
    })

    
    # Initialize environment
    env = VizDoomGym(render=False)
    
    # Create PPO agent
    agent = PPOAgent(
        model, 
        processor, 
        device=device,
        lr=wandb.config.lr if run is not None else 3e-4,
        clip_ratio=wandb.config.clip_ratio if run is not None else 0.2,
        gamma=wandb.config.gamma if run is not None else 0.99,
        lambda_gae=wandb.config.lambda_gae if run is not None else 0.95
    )
    
    # Create PPO trainer
    trainer = PPOTrainer(
        agent, 
        env, 
        batch_size=wandb.config.batch_size if run is not None else 4, 
        epochs=wandb.config.epochs if run is not None else 2, 
        tics=wandb.config.tics if run is not None else 4
    )
    
    # Train the agent
    trainer.train(
        num_timesteps=wandb.config.num_timesteps if run is not None else 200, 
        num_iterations=wandb.config.num_iterations if run is not None else 20,
        save_models=wandb.config.save_models if run is not None else True,
        save_dir=wandb.config.save_dir if run is not None else "./models"
    )
    
    # Close environment
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()