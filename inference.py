import os
import sys
import json
import asyncio
import time

from typing import List, Optional
from openai import AsyncOpenAI
import httpx

from client import MyEnv
from models import EmailAction, ActionType

# Use environment variable for server URL if available, fallback to localhost:7860
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

BENCHMARK = "email_triage"

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

SYSTEM_PROMPT = """You are an Email Triage Assistant. You must manage an inbox consisting of emails.
Available actions:
1. READ <email_id>
2. MOVE <email_id> <folder>
3. REPLY <email_id> <body>
4. FORWARD <email_id> <to_address>
5. SUBMIT

Reply with EXACTLY one action per turn formatted exactly as above. DO NOT output any other text or reasoning.
"""

def parse_model_response(text: str) -> EmailAction:
    text = text.strip()
    parts = text.split(" ")
    command = parts[0].upper()
    try:
        if command == "READ":
            return EmailAction(action_type=ActionType.READ, email_id=parts[1])
        elif command == "MOVE":
            return EmailAction(action_type=ActionType.MOVE, email_id=parts[1], target_folder=parts[2])
        elif command == "REPLY":
            return EmailAction(action_type=ActionType.REPLY, email_id=parts[1], body=" ".join(parts[2:]))
        elif command == "FORWARD":
            return EmailAction(action_type=ActionType.FORWARD, email_id=parts[1], to_address=parts[2])
        elif command == "SUBMIT":
            return EmailAction(action_type=ActionType.SUBMIT)
    except Exception:
        pass
    return EmailAction(action_type=ActionType.SUBMIT)

async def wait_for_server(url: str, timeout: int = 30):
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(f"{url}/health", timeout=1.0)
                if response.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(1.0)
    return False

async def run_task(task_name: str, client: AsyncOpenAI, url: str):
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    env = MyEnv(url)
    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        result = await env.reset(task_name=task_name) 
        
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
                
        for step_idx in range(1, 10):
            if result.done:
                break
                
            steps_taken = step_idx
            
            obs = result.observation
            obs_str = f"Feedback: {obs.system_message}\n"
            if obs.read_email_content:
                obs_str += f"Email Content: {obs.read_email_content}\n"
            
            obs_str += "Inbox:\n"
            for email in obs.inbox_summary:
                obs_str += f"- ID: {email.id} | Sender: {email.sender} | Subject: {email.subject} | Folder: {email.folder}\n"
                
            history.append({"role": "user", "content": obs_str})
            
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=history,
                    temperature=0.0
                )
                action_str = response.choices[0].message.content.strip()
                history.append({"role": "assistant", "content": action_str})
            except Exception as e:
                print(f"API Error at step {step_idx}: {e}", flush=True)
                action_str = "SUBMIT"
                
            action = parse_model_response(action_str)
            
            try:
                result = await env.step(action)
            except Exception as e:
                log_step(step=step_idx, action=action_str, reward=0.0, done=True, error=str(e))
                break
                
            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            
            log_step(step=step_idx, action=action_str, reward=reward, done=done, error=None)
            
            if done:
                score = reward
                break
                
        score = min(max(score, 0.0), 1.0)
        success = score >= 1.0
        
    except Exception as e:
        print(f"CRITICAL ERROR in run_task: {e}", flush=True)
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
    finally:
        try:
            await env.close()
        except Exception:
            pass

async def main():
    print(f"Starting inference with ENV_URL={ENV_URL}, MODEL_NAME={MODEL_NAME}", flush=True)
    
    url = ENV_URL
    try:
        if not await wait_for_server(url):
            print(f"Server at {url} not reachable. Proceeding with caution...", flush=True)
            # We don't return immediately, maybe the health check is just failing but the API works
    except Exception as e:
        print(f"Error during wait_for_server: {e}", flush=True)

    try:
        api_base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        
        # Explicitly configure using environment variables for the LLM proxy check
        client = AsyncOpenAI(
            base_url=api_base_url if api_base_url else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY", "dummy_key")
        )
        
        # We test all 3 tasks sequentially
        for task in ["easy", "medium", "hard"]:
            try:
                await run_task(task, client, url)
            except Exception as task_error:
                print(f"Error running task {task}: {task_error}", flush=True)
    except Exception as e:
        print(f"Unhandled exception in main: {e}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        sys.exit(1) # We still exit non-zero if it's truly fatal, but we logged it
