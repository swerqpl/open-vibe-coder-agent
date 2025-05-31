import openai
import os
import json
import requests
from bs4 import BeautifulSoup, Tag
import time
import subprocess
from dotenv import load_dotenv
import urllib.parse
import platform
import sys
import queue
import threading
import re # DODANY IMPORT RE

# --- Konfiguracja ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Nie znaleziono klucza OPENAI_API_KEY w zmiennych środowiskowych.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
MODEL_PRE_PROCESSOR = "gpt-4o-mini"
MODEL_PLANNER = "gpt-4o"
MODEL_CODER = "gpt-4o-mini" # Użyjmy mini dla kodera, jak w oryginalnym setupie

MAX_ITERATIONS_CODER = 10
MAX_CONSECUTIVE_ERRORS_CODER = 3
MAX_CLARIFICATION_ROUNDS = 3

# --- Funkcje pomocnicze ---

def ask_llm(prompt=None, system_message=None, model=MODEL_PLANNER, temperature=0.7, json_mode=False, messages_override=None):
    messages_to_send = []
    if messages_override is not None:
        messages_to_send = messages_override
    elif prompt is not None:
        if system_message:
            messages_to_send.append({"role": "system", "content": system_message})
        messages_to_send.append({"role": "user", "content": prompt})
    elif system_message is not None:
        messages_to_send.append({"role": "system", "content": system_message})
    else:
        raise ValueError("ask_llm called without prompt, system_message, or non-empty messages_override.")

    if not messages_to_send:
        raise ValueError("No messages to send to LLM. Check inputs for ask_llm.")

    try:
        response_kwargs = {"model": model, "messages": messages_to_send, "temperature": temperature}
        if json_mode:
            response_kwargs["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**response_kwargs)
        content = completion.choices[0].message.content

        if content is None:
            print(f"BŁĄD KRYTYCZNY: API OpenAI (model: {model}) zwróciło 'None' jako zawartość wiadomości dla promptu/messages: {messages_to_send}")
            return None

        if json_mode:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Błąd: LLM (model: {model}) miał zwrócić JSON, ale nie zwrócił poprawnego. Otrzymano:\n{content}")
                # Próba naprawy, jeśli odpowiedź jest otoczona ```json ... ```
                json_match_fallback = re.search(r'```json\s*([\s\S]*?)\s*```', content, re.DOTALL)
                if json_match_fallback:
                    try:
                        return json.loads(json_match_fallback.group(1))
                    except json.JSONDecodeError:
                         print(f"Błąd (po próbie naprawy ```json): LLM (model: {model}) nadal nie zwrócił poprawnego JSON. Otrzymano (po ekstrakcji):\n{json_match_fallback.group(1)}")
                         return None # Lub zwróć None/podnieś błąd, jeśli naprawa też zawiedzie
                return None # Jeśli nie było ```json``` lub naprawa zawiodła
        return content

    except openai.APIError as e:
        print(f"Błąd API OpenAI (model: {model}): {e}")
        return None
    except Exception as e:
        print(f"Błąd podczas komunikacji z OpenAI (model: {model}): {e}")
        return None

def search_duckduckgo(query: str, num_results: int = 10) -> list:
    print(f"INFO: Wyszukiwanie w DuckDuckGo: '{query}' (max {num_results} wyników)")
    if not query:
        return []
    results = []
    try:
        params = {'q': query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get("https://html.duckduckgo.com/html/", params=params, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        potential_results_containers = soup.find_all('div', class_='result__body', limit=(num_results + 5))
        for result_container_maybe in potential_results_containers:
            if not isinstance(result_container_maybe, Tag): continue
            result_container: Tag = result_container_maybe
            title_tag_maybe = result_container.find('a', class_='result__a')
            snippet_tag_maybe = result_container.find('a', class_='result__snippet')
            if isinstance(title_tag_maybe, Tag) and isinstance(snippet_tag_maybe, Tag):
                title_tag: Tag = title_tag_maybe
                snippet_tag: Tag = snippet_tag_maybe
                title = title_tag.get_text(strip=True)
                link_ddg_maybe = title_tag.get('href')
                if isinstance(link_ddg_maybe, str):
                    link_ddg: str = link_ddg_maybe
                    link: str = link_ddg
                    if link_ddg.startswith("/l/"):
                        parsed_url_obj = urllib.parse.urlparse(link_ddg)
                        parsed_link_qs_str = parsed_url_obj.query
                        actual_link_param_list = urllib.parse.parse_qs(parsed_link_qs_str).get('uddg')
                        if actual_link_param_list and isinstance(actual_link_param_list, list) and len(actual_link_param_list) > 0 and isinstance(actual_link_param_list[0], str):
                            link = actual_link_param_list[0]
                        else: continue
                    if not link.startswith("http"): continue
                    snippet = snippet_tag.get_text(strip=True)
                    results.append({"title": title, "link": link, "snippet": snippet})
                    if len(results) >= num_results: break
        print(f"INFO: Zebrano {len(results)} wyników dla '{query}'.")
        return results
    except requests.exceptions.Timeout: print(f"BŁĄD: Timeout podczas wyszukiwania w DuckDuckGo dla frazy: '{query}'"); return []
    except requests.exceptions.RequestException as e: print(f"Błąd RequestException podczas wyszukiwania w DuckDuckGo: {e}"); return []
    except Exception as e_general: print(f"Nieoczekiwany, ogólny błąd w search_duckduckgo podczas parsowania dla '{query}': {e_general}"); return []

def select_relevant_links_llm(search_query, search_results, original_prompt, num_to_select=3):
    print(f"INFO: Wybieranie do {num_to_select} linków dla zapytania: '{search_query}'")
    if not search_results:
        print("INFO: Brak wyników wyszukiwania do przetworzenia przez LLM.")
        return []

    formatted_results = "\n".join([
        f"{idx+1}. Title: {res['title']}\n   Link: {res['link']}\n   Snippet: {res['snippet']}\n"
        for idx, res in enumerate(search_results)
    ])

    prompt_for_llm = f"""
Original user request: "{original_prompt}"
Search query performed: "{search_query}"
Available search results:
{formatted_results}

Based on the original user request and the specific search query, select up to {num_to_select} MOST relevant links from the list above.
Your answer MUST be a JSON list of objects. Each object MUST have "title", "link", and "reason" keys.
- "title": The exact title of the selected search result.
- "link": The exact link of the selected search result.
- "reason": A brief explanation (1-2 sentences) of why this link is highly relevant.

**CRITICAL: You MUST select at least ONE link if there are any search results provided and you believe at least one could be even remotely useful.** If none seem perfectly relevant, select up to {num_to_select} (or fewer, if less than {num_to_select} results are available or relevant) that are the *closest match* or *most likely to contain useful information*, even if indirectly. Explain your choice in the "reason".
If truly NO links are even remotely useful, you may return an empty JSON list [], but this should be rare.

Ensure your entire response is ONLY a single, valid JSON list of objects, like this example:
[
  {{"title": "Example Title 1", "link": "http://example.com/1", "reason": "Directly addresses part of the query regarding X."}},
  {{"title": "Example Title 2", "link": "http://example.com/2", "reason": "Offers background information on Y, which is related to the query."}}
]
Do NOT include any other text, explanations, or markdown formatting outside of the JSON list itself.
If there are fewer than {num_to_select} search results provided in total, you may select all of them if they are relevant.
"""
    system_message = "You are an expert research assistant. Your output MUST be a valid JSON list of objects as specified, and nothing else. Adhere strictly to the formatting rules."

    selected_links_json_str = ask_llm(prompt=prompt_for_llm, system_message=system_message, model=MODEL_PLANNER, json_mode=False, temperature=0.3)

    selected_links_json = None
    if selected_links_json_str:
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', selected_links_json_str, re.DOTALL)
            if json_match:
                json_content = json_match.group(1) if json_match.group(1) else json_match.group(2)
                selected_links_json = json.loads(json_content)
            else:
                selected_links_json = json.loads(selected_links_json_str)
        except json.JSONDecodeError:
            print(f"Błąd: LLM (select_relevant_links) nie zwrócił poprawnego JSON-a. Otrzymano:\n{selected_links_json_str}")
            error_prompt_json_fix = f"The following text was supposed to be a valid JSON list of objects, but it's malformed. Please reformat it into a single, valid JSON list of objects. Each object should have 'title', 'link', and 'reason' keys. Do not add any explanations, just the corrected JSON list.\n\nMalformed text:\n```\n{selected_links_json_str}\n```"

            fixed_json_str = ask_llm(prompt=error_prompt_json_fix, system_message="You are a JSON correction expert. Respond ONLY with the corrected JSON.", model=MODEL_PRE_PROCESSOR, temperature=0.1, json_mode=False)
            if fixed_json_str:
                try:
                    json_match_retry = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', fixed_json_str, re.DOTALL)
                    if json_match_retry:
                         json_content_retry = json_match_retry.group(1) if json_match_retry.group(1) else json_match_retry.group(2)
                         selected_links_json = json.loads(json_content_retry)
                    else:
                         selected_links_json = json.loads(fixed_json_str)
                    print(f"INFO: JSON naprawiony przez LLM (2 próba).")
                except json.JSONDecodeError:
                    print(f"Błąd krytyczny: LLM dwukrotnie nie zwrócił/naprawił JSON-a (select_relevant_links). Otrzymano (2 próba naprawy):\n{fixed_json_str}")
                    selected_links_json = None
            else:
                print(f"Błąd krytyczny: LLM nie odpowiedział na prośbę o naprawę JSON (select_relevant_links).")
                selected_links_json = None
    else:
        print(f"Błąd: ask_llm zwróciło None dla select_relevant_links.")


    if selected_links_json and isinstance(selected_links_json, list):
        valid_links = []
        for item in selected_links_json:
            if isinstance(item, dict) and "title" in item and "link" in item and "reason" in item:
                valid_links.append(item)
            else:
                print(f"OSTRZEŻENIE: LLM (select_relevant_links) zwrócił niekompletny obiekt linku: {item}")

        if len(valid_links) > 0:
            final_selection = valid_links[:num_to_select]
            print(f"INFO: LLM wybrał {len(final_selection)} poprawnych linków.")
            return final_selection
        elif len(selected_links_json) == 0:
             print(f"INFO: LLM świadomie zwrócił pustą listę linków.")
             return []
        else:
            print(f"INFO: LLM nie wybrał żadnych poprawnych linków, mimo że zwrócił listę z błędnymi obiektami. Używam fallback.")
            return [{"title": r["title"], "link": r["link"], "reason": f"Fallback: LLM failed to select valid links. Result #{i+1} from search."}
                    for i, r in enumerate(search_results[:num_to_select])]
    else:
        reason_for_fallback = "LLM failure or invalid format (e.g. returned None, non-list, or failed to parse)."
        if selected_links_json is None and selected_links_json_str is not None :
             reason_for_fallback = "LLM returned malformed JSON that could not be repaired."
        elif selected_links_json_str is None:
            reason_for_fallback = "LLM (ask_llm) returned no response for link selection."

        print(f"INFO: {reason_for_fallback} Używam fallback.")
        return [{"title": r["title"], "link": r["link"], "reason": f"Fallback: {reason_for_fallback} Result #{i+1} from search."}
                for i, r in enumerate(search_results[:num_to_select])]

# ZMODYFIKOWANA FUNKCJA
def execute_python_code(code_string: str, working_directory: str, file_name: str, execute_flag: bool = True):
    """
    Zapisuje podany kod Python do pliku w specificznym katalogu roboczym,
    a następnie opcjonalnie wykonuje ten plik. Plik NIE jest usuwany po wykonaniu.
    """
    print(f"INFO: Przygotowywanie kodu Python w '{working_directory}' (plik: {file_name}, wykonać: {execute_flag})")
    # print(f"Kod:\n```python\n{code_string}\n```") # Można odkomentować dla pełnego logu kodu

    if not file_name: # Dodatkowe zabezpieczenie, choć powinno być wyłapane wcześniej
        error_msg = "BŁĄD KRYTYCZNY: Nazwa pliku dla execute_python_code nie może być pusta."
        print(error_msg)
        return "", error_msg, -111

    full_file_path = os.path.join(working_directory, file_name)

    # Zabezpieczenie przed próbą zapisu poza katalogiem roboczym
    abs_working_dir = os.path.abspath(working_directory)
    abs_file_path = os.path.abspath(full_file_path)

    if not abs_file_path.startswith(abs_working_dir) or abs_file_path == abs_working_dir:
        error_msg = (f"BŁĄD KRYTYCZNY: Wykryto próbę zapisu/wykonania pliku Python ('{file_name}') "
                     f"poza dozwolonym katalogiem roboczym ('{working_directory}') lub jako sam katalog. Operacja zablokowana.")
        print(error_msg)
        return "", error_msg, -110

    try:
        os.makedirs(working_directory, exist_ok=True)
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(code_string)
        print(f"INFO: Kod Python zapisany do pliku: {full_file_path}")

        if not execute_flag:
            print(f"INFO: Plik '{file_name}' został zapisany. Wykonanie pominięte zgodnie z flagą 'execute: false'.")
            return "", "Plik zapisany, wykonanie pominięte.", 0

        python_executable_to_use = sys.executable
        venv_name = "my_agent_venv"
        if platform.system() == "Windows":
            potential_venv_python_path = os.path.join(working_directory, venv_name, "Scripts", "python.exe")
        else:
            potential_venv_python_path = os.path.join(working_directory, venv_name, "bin", "python")

        if os.path.exists(potential_venv_python_path) and os.access(potential_venv_python_path, os.X_OK):
            python_executable_to_use = potential_venv_python_path
            print(f"INFO: Znaleziono i zostanie użyty interpreter Python z projektu venv: {python_executable_to_use}")
        else:
            print(f"INFO: Nie znaleziono interpretera Python w projekcie venv ('{potential_venv_python_path}'). Używam systemowego/bootstrapowego interpretera: {sys.executable}")
            print(f"      Jeśli kod wymaga specjalnych zależności, mogły one nie zostać poprawnie zainstalowane lub venv nie został utworzony/znaleziony.")
        
        print(f"INFO: Wykonywanie skryptu Python '{file_name}' za pomocą '{python_executable_to_use}'...")
        process = subprocess.Popen(
            [python_executable_to_use, "-u", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=working_directory
        )
        stdout, stderr = process.communicate(timeout=600)
        return_code = process.returncode

        print(f"INFO: Skrypt Python '{file_name}' (uruchomiony przez {os.path.basename(python_executable_to_use)}) zakończony z kodem: {return_code}")
        if stdout: print(f"STDOUT ({file_name}):\n{stdout}")
        if stderr: print(f"STDERR ({file_name}):\n{stderr}")
        return stdout, stderr, return_code

    except subprocess.TimeoutExpired:
        interpreter_name_for_log = python_executable_to_use if 'python_executable_to_use' in locals() else sys.executable
        print(f"BŁĄD: Timeout podczas wykonywania skryptu Python '{file_name}' przez {os.path.basename(interpreter_name_for_log)}.")
        return "", f"TimeoutExpired: Process for '{file_name}' killed after 600 seconds.", -101
    except FileNotFoundError:
        interpreter_path = python_executable_to_use if 'python_executable_to_use' in locals() else "nieokreślony interpreter"
        print(f"BŁĄD: Nie znaleziono interpretera Python: {interpreter_path} podczas próby uruchomienia '{file_name}'.")
        return "", f"FileNotFoundError: Interpreter {interpreter_path} not found (when trying to run '{file_name}').", -102
    except Exception as e:
        print(f"BŁĄD: Nieoczekiwany wyjątek podczas zapisywania/wykonywania skryptu Python '{file_name}': {e}")
        return "", str(e), -103
    # Blok finally z os.remove() został celowo usunięty.

def _enqueue_output_execute_terminal_command(stream, q, stream_name_tag, stop_event):
    try:
        for line in iter(stream.readline, ''):
            if stop_event.is_set():
                break
            q.put((stream_name_tag, line))
    except ValueError:
        pass
    except Exception as e:
        try:
            q.put((stream_name_tag, f"THREAD_EXCEPTION: {str(e)}\n"))
        except Exception:
            pass
    finally:
        try:
            stream.close()
        except Exception:
            pass
        try:
            q.put((stream_name_tag, None))
        except Exception:
            pass

def execute_terminal_command(command, working_directory, output_silence_timeout=120, overall_timeout=300):
    print(f"INFO: Wykonywanie komendy terminala w '{working_directory}': `{command}` (output_silence_timeout={output_silence_timeout}s, overall_timeout={overall_timeout}s)")
    stdout_parts = []
    stderr_parts = []
    return_code = None

    process = None
    stop_threads_event = threading.Event()
    stdout_thread = None
    stderr_thread = None

    start_time = time.monotonic()

    try:
        os.makedirs(working_directory, exist_ok=True)
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=working_directory,
            bufsize=1
        )

        q = queue.Queue()

        stdout_thread = threading.Thread(target=_enqueue_output_execute_terminal_command, args=(process.stdout, q, "stdout", stop_threads_event))
        stderr_thread = threading.Thread(target=_enqueue_output_execute_terminal_command, args=(process.stderr, q, "stderr", stop_threads_event))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        last_output_time = time.monotonic()
        stdout_thread_finished = False
        stderr_thread_finished = False

        while not (stdout_thread_finished and stderr_thread_finished):
            if process.poll() is not None and stdout_thread_finished and stderr_thread_finished:
                break

            try:
                stream_name, line = q.get(timeout=0.2)

                if line is None:
                    if stream_name == "stdout": stdout_thread_finished = True
                    else: stderr_thread_finished = True
                else:
                    last_output_time = time.monotonic()
                    if stream_name == "stdout": stdout_parts.append(line)
                    else: stderr_parts.append(line)

            except queue.Empty:
                if process.poll() is not None and stdout_thread_finished and stderr_thread_finished:
                    break

                if time.monotonic() - start_time > overall_timeout:
                    print(f"BŁĄD: Ogólny Timeout! Proces przekroczył {overall_timeout}s. Przerywanie procesu.")
                    stop_threads_event.set()
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if process.poll() is None: process.kill(); process.wait()
                    stderr_parts.append(f"\nOverallTimeoutError: Proces zabity po przekroczeniu {overall_timeout} sekund.")
                    return_code = -201
                    break

                if time.monotonic() - last_output_time > output_silence_timeout:
                    print(f"BŁĄD: Timeout ciszy! Brak nowego outputu przez {output_silence_timeout}s. Przerywanie procesu.")
                    stop_threads_event.set()
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if process.poll() is None: process.kill(); process.wait()
                    stderr_parts.append(f"\nOutputSilenceTimeoutError: Proces zabity po {output_silence_timeout} sekundach braku nowego outputu.")
                    return_code = -202
                    break

        stop_threads_event.set()

        if stdout_thread is not None and stdout_thread.is_alive(): stdout_thread.join(timeout=2)
        if stderr_thread is not None and stderr_thread.is_alive(): stderr_thread.join(timeout=2)

        while not q.empty():
            try:
                stream_name, line = q.get_nowait()
                if line is None: continue
                if stream_name == "stdout": stdout_parts.append(line)
                else: stderr_parts.append(line)
            except queue.Empty: break

        if return_code is None:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, overall_timeout - (time.monotonic() - start_time)))
                except subprocess.TimeoutExpired:
                     print(f"OSTRZEŻENIE: Proces nie zakończył się w wyznaczonym czasie po zakończeniu pętli odczytu. Używam kill().")
                     if process.poll() is None: process.kill(); process.wait()

            return_code = process.returncode
            if return_code is None:
                 print(f"OSTRZEŻENIE KRYTYCZNE: return_code jest None, mimo że proces.poll() to {process.poll()}. Ustawiam na -203.")
                 return_code = -203

        stdout_str = "".join(stdout_parts)
        stderr_str = "".join(stderr_parts)

        print(f"INFO: Komenda terminala zakończona z kodem: {return_code}")
        if stdout_str: print(f"STDOUT:\n{stdout_str}")
        if stderr_str: print(f"STDERR:\n{stderr_str}")

        return stdout_str, stderr_str, return_code

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono komendy lub nieprawidłowa ścieżka przy Popen: {command.split()[0] if command else 'pusta komenda'}")
        return "", f"FileNotFoundError (Popen): Command or part of command not found: {command}", -204
    except Exception as e:
        print(f"BŁĄD: Nieoczekiwany wyjątek globalny podczas wykonywania komendy terminala: {e}")
        stdout_str = "".join(stdout_parts)
        stderr_str = "".join(stderr_parts) + f"\nGlobalException: {str(e)}"
        if process and process.poll() is None:
            stop_threads_event.set()
            try: process.kill(); process.wait(timeout=2)
            except Exception as kill_e: print(f"BŁĄD podczas próby zabicia procesu w bloku wyjątku: {kill_e}")
        return stdout_str, stderr_str, -205
    finally:
        if process:
            if process.stdout and not process.stdout.closed:
                try: process.stdout.close()
                except Exception: pass
            if process.stderr and not process.stderr.closed:
                try: process.stderr.close()
                except Exception: pass

            if not stop_threads_event.is_set():
                stop_threads_event.set()

            if stdout_thread and stdout_thread.is_alive():
                try: stdout_thread.join(timeout=1)
                except Exception: pass
            if stderr_thread and stderr_thread.is_alive():
                try: stderr_thread.join(timeout=1)
                except Exception: pass

            if process.poll() is None:
                print("OSTRZEŻENIE: Proces wciąż aktywny w bloku 'finally'. Próba kill().")
                try: process.kill(); process.wait(timeout=2)
                except Exception as final_kill_e: print(f"BŁĄD podczas ostatecznej próby zabicia procesu: {final_kill_e}")


def clarify_initial_task_with_user(initial_user_prompt: str, project_working_dir_info: str, is_continuation: bool = False) -> str:
    print("\n--- FAZA DOPRECYZOWANIA ZADANIA Z UŻYTKOWNIKIEM ---")
    continuation_context_str = ""
    if is_continuation:
        continuation_context_str = (
            f"This is a follow-up request for an existing project located in '{project_working_dir_info}'. "
            "The user has already completed a previous task (or tasks) in this project directory "
            "and now wants to add more functionality or make changes. "
            "Please consider this context when clarifying their new request. "
            "You can assume the project directory might already contain files, a Python virtual environment ('my_agent_venv'), etc. "
            "Focus your questions on the *new* requirements."
        )

    system_message_pre_processor = f"""
You are an AI assistant whose role is to clarify a user's request for a larger AI agent.
{continuation_context_str}
The larger AI agent will then attempt to fulfill this clarified request.
The agent will be working on a {platform.system()} system.
The agent's dedicated project working directory is: "{project_working_dir_info}". It will create all files and subdirectories (like 'my_agent_venv' for Python dependencies) there.

Your goal is to:
1. Analyze the user's current request.
2. Identify any ambiguities, missing information, or potential issues that could hinder the main AI agent for THIS specific request.
3. Formulate concise questions for the user to gather necessary details FOR THIS REQUEST.
4. Based on the user's answers, refine the request into a more precise and actionable task description for the main AI agent.

Interaction flow:
- You will receive the user's current request.
- You will respond with a JSON object. This JSON object MUST have two keys:
  - "questions_for_user": A list of strings. Each string is a question for the user. If no questions are needed for this request, this list should be empty.
  - "current_refined_task": A string representing your current understanding of THIS task. This should be updated based on user answers.

- If "questions_for_user" is not empty, the user will provide answers, and you will receive them in the next turn to further refine the task.
- If "questions_for_user" is empty, it means you believe the task is clear enough. The "current_refined_task" will then be used as the final input for the main AI agent for this cycle.
Ensure your entire response is a single, valid JSON object.
"""
    conversation_history = []
    current_task_description = initial_user_prompt
    for i in range(MAX_CLARIFICATION_ROUNDS + 1):
        print(f"\nPre-procesor: Runda {i+1}/{MAX_CLARIFICATION_ROUNDS+1}")
        if i == 0:
            user_input_for_llm = f"User's current request: \"{initial_user_prompt}\"\n\nBased on this, what is your current understanding and what questions do you have for the user regarding THIS request? Remember to provide your response as a JSON object with 'questions_for_user' and 'current_refined_task'."
            conversation_history.append({"role": "user", "content": user_input_for_llm})
        llm_messages = [{"role": "system", "content": system_message_pre_processor}]
        llm_messages.extend(conversation_history)
        llm_messages.append({"role": "user", "content": "Provide your analysis and any questions as a JSON object with 'questions_for_user' and 'current_refined_task'."})
        response_json = ask_llm(messages_override=llm_messages, model=MODEL_PRE_PROCESSOR, json_mode=True, temperature=0.4)
        if not response_json or not isinstance(response_json, dict) or \
           "questions_for_user" not in response_json or \
           "current_refined_task" not in response_json:
            print(f"BŁĄD: Bot pre-procesujący nie zwrócił poprawnego JSON-a. Otrzymano: {response_json}")
            print("Używam oryginalnego zadania.")
            return initial_user_prompt
        conversation_history.append({"role": "assistant", "content": json.dumps(response_json)})
        questions = response_json.get("questions_for_user", [])
        current_task_description = response_json.get("current_refined_task", initial_user_prompt)
        print(f"Aktualne rozumienie zadania przez bota: \"{current_task_description}\"")
        if not questions or i >= MAX_CLARIFICATION_ROUNDS:
            if not questions and i < MAX_CLARIFICATION_ROUNDS: print("Bot pre-procesujący uznał zadanie za wystarczająco jasne.")
            elif questions and i >= MAX_CLARIFICATION_ROUNDS: print(f"Osiągnięto maksymalną liczbę rund doprecyzowania ({MAX_CLARIFICATION_ROUNDS}). Używam ostatniej wersji zadania.")
            break
        user_answers_text = ""
        print("Bot pre-procesujący ma następujące pytania:")
        for q_idx, question_text in enumerate(questions):
            answer = ""
            while not answer.strip():
                answer = input(f"  Pytanie {q_idx+1}: {question_text}\n  Twoja odpowiedź: > ").strip()
                if not answer: print("Odpowiedź nie może być pusta. Spróbuj ponownie.")
            user_answers_text += f"Q: {question_text}\nA: {answer}\n"
        conversation_history.append({"role": "user", "content": f"Here are my answers to your previous questions:\n{user_answers_text}\nPlease refine the task based on these answers and ask more questions if necessary."})
    print(f"\n--- Zakończono fazę doprecyzowania zadania ---")
    print(f"Ostateczne, doprecyzowane zadanie dla agenta: \"{current_task_description}\"")
    return current_task_description

class AIAgent:
    def __init__(self, initial_task: str, project_working_directory: str):
        self.initial_task = initial_task
        self.project_working_directory = os.path.abspath(project_working_directory)
        self.planner_history = []
        self.coder_history = []
        self.overall_plan = None
        self.researched_info = []
        self.resuming_coder_after_human_input = False
        self.working_directory = self.project_working_directory
        try:
            os.makedirs(self.working_directory, exist_ok=True)
            history_subdir = os.path.join(self.working_directory, "agent_history")
            os.makedirs(history_subdir, exist_ok=True)
            print(f"INFO: Agent będzie pracował w dedykowanym folderze projektu: {self.working_directory}")
            print(f"INFO: Historia pracy agenta będzie zapisywana w podfolderze: {history_subdir}")
        except OSError as e:
            print(f"KRYTYCZNY BŁĄD: Nie można utworzyć/uzyskać dostępu do folderu roboczego projektu '{self.working_directory}' lub podfolderu historii: {e}")
            raise
        self.platform_system = platform.system()

    def reset_for_new_task(self, new_initial_task: str):
        print(f"INFO: Resetowanie agenta dla nowego zadania: \"{new_initial_task}\"")
        self.initial_task = new_initial_task
        self.planner_history = []
        self.coder_history = []
        self.overall_plan = None
        self.resuming_coder_after_human_input = False

    def _add_to_history(self, history_list, role, content):
        history_list.append({"role": role, "content": content})

    def planner_phase(self):
        print("\n--- FAZA PLANOWANIA ---")
        context_summary = f"Platform: {self.platform_system}. Working Directory: '{self.working_directory}'. "
        context_summary += f"Python venv (if needed): 'my_agent_venv' inside working directory. "
        if self.researched_info:
            context_summary += f"Previously researched info (last 5 items, may or may not be relevant to current task): {json.dumps(self.researched_info[-5:])}. "
        else:
            context_summary += "No information from previous research phases is currently stored. "

        system_msg_planner_initial = (
            f"You are a meticulous AI planner. Your primary task is to break down a complex, user-defined programming or creation task into a high-level, actionable plan. "
            f"Additionally, you must identify 0-2 CRITICAL search queries (in English) if, and ONLY IF, gathering external information is absolutely essential for creating a sensible plan FOR THE CURRENT TASK. "
            f"The user's request has already been clarified. Your focus is now on strategic planning for this specific request. "
            f"Task context: {context_summary}\n"
            f"The project directory '{self.working_directory}' might already contain files from previous tasks. Your plan should account for this possibility (e.g., modifying existing files, adding new ones, checking existing structure with LIST_FILES if necessary for planning). "
            f"Key considerations for your plan and search queries:\n"
            f"- Dependencies: Does the task imply the use of specific external libraries, APIs, or tools that need to be installed or accessed? Consider if they might already be in 'my_agent_venv'.\n"
            f"- Algorithms: Are there complex algorithms or non-trivial logic patterns required that might benefit from examples or documentation?\n"
            f"- Data Formats/Sources: If external data is involved, are its format and source clear? If not, a search query MIGHT be needed IF it's critical for planning.\n"
            f"- Feasibility: Briefly assess if the task seems achievable with standard tools. If it seems extremely complex or requires very niche knowledge, this might influence your plan and search queries.\n"
            f"DO NOT generate search queries for:\n"
            f"  - Basic Python syntax or standard library usage.\n"
            f"  - General 'how-to' for common tasks unless they involve a very specific, non-obvious technology not covered by existing research.\n"
            f"  - How to create or activate Python virtual environments (the agent has built-in instructions for this).\n"
            f"Output a JSON object with two keys:\n"
            f"1. 'plan_steps': A list of strings. Each string is a clear, logical step in the plan. These steps should be granular enough for a Coder AI to implement but high-level enough to represent a strategic breakdown. Aim for 3-7 major steps.\n"
            f"2. 'search_queries': A list of 0-2 highly specific search query strings (in English). Only include queries if the information is VITAL for planning and cannot be reasonably assumed or found in existing research. If no searches are VITAL, provide an empty list [].\n"
            f"Ensure your output is a single, valid JSON object and nothing else.\n"
        )
        prompt_planner_initial = f"Clarified user request for the current task: \"{self.initial_task}\"\n\nBased on this request and the provided context (including potential existing project state and prior research), generate the plan and any critical search queries for THIS task."
        llm_messages = [{"role": "system", "content": system_msg_planner_initial}, {"role": "user", "content": prompt_planner_initial}]
        response_json = ask_llm(messages_override=llm_messages, model=MODEL_PLANNER, json_mode=True, temperature=0.3)
        if not response_json or not isinstance(response_json, dict) or "plan_steps" not in response_json:
            print(f"BŁĄD: Nie udało się wygenerować planu przez planistę. Otrzymano: {response_json}")
            self._add_to_history(self.planner_history, "user", prompt_planner_initial)
            self._add_to_history(self.planner_history, "assistant", f"Invalid JSON response: {json.dumps(response_json)}")
            return False
        self.overall_plan = response_json["plan_steps"]
        search_queries = response_json.get("search_queries", [])
        self._add_to_history(self.planner_history, "user", prompt_planner_initial)
        self._add_to_history(self.planner_history, "assistant", json.dumps(response_json))
        print(f"INFO: Wstępny plan ({len(self.overall_plan)} kroków) dla bieżącego zadania:")
        for i, step in enumerate(self.overall_plan): print(f"  {i+1}. {step}")
        print(f"INFO: Zapytania do wyszukiwarki: {search_queries if search_queries else 'Brak'}")
        if search_queries:
            print("\nINFO: Rozpoczynanie fazy wyszukiwania informacji dla bieżącego zadania...")
            for query in search_queries:
                if not query.strip(): continue
                raw_results = search_duckduckgo(query, num_results=7)
                if raw_results:
                    selected_links = select_relevant_links_llm(query, raw_results, self.initial_task, num_to_select=3)
                    if selected_links:
                        print(f"  Wybrane linki dla zapytania '{query}':")
                        if not selected_links:
                             print("    (LLM nie wybrał żadnych linków dla tego zapytania)")
                        for link_info in selected_links:
                            new_info = f"Title: {link_info['title']}, Link: {link_info['link']}, Reason: {link_info.get('reason', 'N/A')}, Related_Query_Current_Task: '{query}'"
                            if not any(existing_info.startswith(f"Title: {link_info['title']}, Link: {link_info['link']}") for existing_info in self.researched_info):
                                self.researched_info.append(new_info)
                            print(f"    - {link_info['title']} ({link_info['link']})")
                else: print(f"  Brak wyników dla zapytania '{query}'.")
                time.sleep(0.5)
            print(f"\nINFO: Zakończono zbieranie informacji. Całkowita liczba zebranych elementów (łącznie z poprzednimi zadaniami): {len(self.researched_info)}")
            if self.researched_info:
                print("Ostatnio zebrane/istniejące informacje (do 5 ostatnich pozycji):")
                for item in self.researched_info[-5:]: print(f"- {item}")
            elif search_queries: print("OSTRZEŻENIE: Nie udało się zebrać żadnych informacji z wyszukiwarki, mimo wygenerowanych zapytań dla bieżącego zadania.")
        return True

    def coder_phase_loop(self, current_sub_task_prompt: str):
        print(f"\n--- FAZA KODOWANIA (Pętla) ---")
        print(f"Aktualne zadanie dla Kodera: {current_sub_task_prompt}")
        previous_error_signature = None
        consecutive_error_count = 0
        for attempt in range(MAX_ITERATIONS_CODER):
            print(f"\nKoder: Próba {attempt + 1}/{MAX_ITERATIONS_CODER} dla bieżącego podzadania. (Błędów z rzędu: {consecutive_error_count})")
            researched_info_summary = "No relevant information from research phase seems directly applicable to this specific sub-task, or no research was performed / is currently stored."
            if self.researched_info:
                keywords_from_subtask = [kw for kw in current_sub_task_prompt.lower().split() if len(kw) > 3]
                relevant_researched = []
                if keywords_from_subtask:
                    relevant_researched = [
                        info for info in self.researched_info
                        if any(kw in info.lower() for kw in keywords_from_subtask)
                    ]
                if not relevant_researched and self.researched_info:
                    relevant_researched = self.researched_info[-3:]

                if relevant_researched:
                    researched_info_summary = "Potentially relevant information from research phase (use your judgment; this includes info from previous tasks in this project):\n"
                    researched_info_summary += "\n".join([f"- {info}" for info in relevant_researched])
                else:
                    researched_info_summary = "Some general research was done (possibly for previous tasks), but no specific items seem directly relevant to this sub-task. You can request a 'SEARCH' if new information is needed."

            venv_name = "my_agent_venv"
            venv_path_windows = os.path.join(self.working_directory, venv_name, "Scripts")
            venv_path_unix = os.path.join(self.working_directory, venv_name, "bin")

            venv_instructions = f"If Python dependencies are needed, they MUST be installed and used within a virtual environment named '{venv_name}' created in the root of the current project working directory ('{self.working_directory}').\n"
            if self.platform_system == "Windows":
                venv_instructions += f"- To create venv (if it doesn't exist): `python -m venv \"{os.path.join(self.working_directory, venv_name)}\"` (execute this via TERMINAL action. Note the quotes if path contains spaces).\n"
                venv_instructions += f"- To install packages: `\"{os.path.join(venv_path_windows, 'pip.exe')}\" install package_name`.\n"
                venv_instructions += f"- To run a script with venv: `\"{os.path.join(venv_path_windows, 'python.exe')}\" script_name.py`.\n"
            else:
                venv_instructions += f"- To create venv (if it doesn't exist): `{sys.executable} -m venv \"{os.path.join(self.working_directory, venv_name)}\"` (execute via TERMINAL action. Note the quotes if path contains spaces).\n"
                venv_instructions += f"- To install packages: `\"{os.path.join(venv_path_unix, 'pip')}\" install package_name`.\n"
                venv_instructions += f"- To run a script with venv: `\"{os.path.join(venv_path_unix, 'python')}\" script_name.py`.\n"
            venv_instructions += f"IMPORTANT: Commands like 'activate' via `TERMINAL` action do NOT persistently activate the venv for subsequent actions in this agent. ALWAYS use full paths to venv executables (pip, python) as shown above for installing packages or running scripts that need the venv."

            venv_management_instructions = f"""
    **Managing '{venv_name}':**
    - **If your sub-task is to 'set up' or 'create' '{venv_name}'**:
        1. First, use 'LIST_FILES' for the root working directory ('{self.working_directory}') to check if a directory named '{venv_name}' already exists.
        2. **If '{venv_name}' already exists**:
            a. Assume it might be corrupted or from a previous failed attempt OR from a previous successful task in this project.
            b. If the goal is to ensure a *fresh* venv for *this specific sub-task's needs* (e.g., planner explicitly stated to recreate it), then use a 'TERMINAL' action to REMOVE the existing '{venv_name}' directory. Example for Unix-like: `rm -rf \"{os.path.join(self.working_directory, venv_name)}\"`. For Windows: `rd /s /q \"{os.path.join(self.working_directory, venv_name)}\"`. Use the correct command for the current platform ({self.platform_system}).
            c. After successful removal (or if it didn't exist, or if you decided not to remove it because it's likely fine from a previous task), attempt to create it fresh using the standard venv creation command if it doesn't exist or was just removed. Note any errors from removal/creation.
        3. **If '{venv_name}' does not exist**:
            a. Proceed to create it fresh using the standard venv creation command specified in 'Virtual Environment (venv) Instructions' above.
        4. After attempting creation (if it was created in this step), verify its existence and basic structure (e.g., check for '{os.path.join(venv_name, 'bin', 'pip')}' on Unix or '{os.path.join(venv_name, 'Scripts', 'pip.exe')}' on Windows using LIST_FILES on the '{venv_name}' directory). If creation fails with an error, analyze the error. If it fails silently (no error, but venv not created properly or key executables are missing), report this to the human.
    - **If your sub-task involves *using* an existing '{venv_name}' (e.g., installing packages, running scripts):**
        - Check its existence with `LIST_FILES` on `{self.working_directory}`. If it's missing, the venv setup likely failed in a previous step or was never part of the plan. You might need to create it (if appropriate for the sub-task and plan) or inform the human.
        - If commands using its executables (like pip or python from the venv) fail with 'file not found' or similar, indicating a broken or missing venv, then the venv setup likely failed or is corrupted. You might need to inform the human or re-evaluate the venv setup strategy, possibly by trying to recreate it if that aligns with the overall plan and current sub-task.
    """
            # ZMODYFIKOWANY system_msg_coder
            system_msg_coder = f"""
You are an AI Coder. Your goal is to meticulously accomplish the given sub-task, which is part of a larger project.
You operate on a {self.platform_system} system.
The ABSOLUTE current working directory for ALL your operations (creating files, running commands) is: "{self.working_directory}". All relative paths you use will be based on this. This directory may already contain files and a '{venv_name}' from previous tasks in this project.

Overall Project Goal (for the current session/user request): "{self.initial_task}"
Current High-Level Plan (from Planner for THIS task): {json.dumps(self.overall_plan)}
Current Specific Sub-Task assigned by the Planner/Project Manager: "{current_sub_task_prompt}"

{researched_info_summary}

Virtual Environment (venv) Instructions for '{venv_name}' (to be created in '{self.working_directory}' if not already suitable):
{venv_instructions}

{venv_management_instructions}

Respond in JSON format with "action_type", "action_details", "reasoning".
Available Action Types:
- "PYTHON": "action_details": JSON object. "reasoning": Your thought process.
  The JSON object for "action_details" MUST have the following keys:
    - "code": A string containing the Python code to be written.
    - "filename": A string specifying the name for the .py file (e.g., "my_script.py").
      This filename MUST end with '.py' and MUST NOT contain any path separators (like '/' or '\\').
      It defines the name of the file to be created (or overwritten if it exists) 
      in the root of the current project working directory: "{self.working_directory}".
      The filename should consist of letters, numbers, underscores, hyphens, and periods only.
    - "execute": (Optional) A boolean value (true or false). Defaults to 'true' if not provided.
      If 'true', the Python script will be executed after being written to the file.
      If 'false', the script will only be written to the file and not executed by this action.
  The Python file specified by "filename" WILL REMAIN in the working directory after this action, regardless of the "execute" flag.
  Example (create and run):
  {{"action_type": "PYTHON", "action_details": {{"code": "print('Hello from file!')", "filename": "hello_world.py", "execute": true}}, "reasoning": "To create and run a simple hello world script that prints a message."}}
  Example (create/write only, no execution):
  {{"action_type": "PYTHON", "action_details": {{"code": "def my_utility_function():\\n  return 'utility result'", "filename": "utils.py", "execute": false}}, "reasoning": "To create a utility file with a helper function. This file is not meant to be run directly but imported by other scripts."}}
- "TERMINAL": "action_details": Command string. "reasoning": Your thought process. For commands that might take a long time (e.g. 'npm install', 'create-react-app'), you can optionally include "suggested_overall_timeout": <seconds> in your JSON, e.g., {{"action_type": "TERMINAL", "action_details": "npm install", "suggested_overall_timeout": 600, "reasoning": "npm install can be slow."}}. Default overall_timeout is 300s.
- "SEARCH": "action_details": Specific search query. "reasoning": Why this search is crucial for THIS sub-task.
- "LIST_FILES": "action_details": (Optional) Sub-path relative to '{self.working_directory}'. If empty, lists CWD. "reasoning": Why listing is necessary.
- "ASK_HUMAN": "action_details": {{"question_for_human": "Your concise question."}}. Use EXTREMELY SPARINGLY. "reasoning": Why human input is essential.
- "FINISHED_SUBTASK": "action_details": Summary of achievement. "reasoning": Confirmation.
- "CANNOT_COMPLETE_SUBTASK": "action_details": Explanation of failure. "reasoning": Justification.

Guidance:
- Analyze conversation history. If a previous attempt for THIS SUB-TASK failed, DO NOT repeat the exact same failing action. Analyze error, fix, or change approach.
- Break down complex operations. Ensure single, valid JSON response. Be precise with paths.
- **When using `TERMINAL` action, prefer to execute single, simple commands.** If you need to execute multiple commands sequentially where the success of one depends on the previous, break them into separate `TERMINAL` actions. Avoid using `&&` or `||` directly in the command string if possible.
- **To read the content of a file, use a `PYTHON` action** with Python's file reading capabilities. `LIST_FILES` is only for listing directory contents.
- **When generating Python code for JSON, use `json.dump()` or `json.dumps()`**.
- **For tasks requiring Python dependencies:**
  1. Follow the '{venv_name}' management instructions above to ensure a functional venv. If it exists and seems fine from a previous task, you may not need to recreate it.
  2. Then, use TERMINAL action to install all required packages into this '{venv_name}' using its specific pip executable.
  3. Finally, use the PYTHON action for code using these dependencies (scripts will be executed using the venv's Python if found).
- If a 'TERMINAL' action for 'pip install' fails:
  - Examine STDERR. If system libraries are missing (e.g., 'portaudio.h not found'), use 'ASK_HUMAN' to request system-level installation. Do NOT attempt 'brew' or 'apt-get' yourself.
- **For Python code needing `input()`:** Use "ASK_HUMAN" first to get values, then use them as variables/constants in your Python code.
"""
            llm_messages = [{"role": "system", "content": system_msg_coder}]
            llm_messages.extend(self.coder_history)
            llm_messages.append({"role": "user", "content": f"Based on the current sub-task: \"{current_sub_task_prompt}\", your available actions, and the conversation history (including any prior work in '{self.working_directory}'), what is your next action? Provide your response as a JSON object."})
            action_json = ask_llm(messages_override=llm_messages, model=MODEL_CODER, json_mode=True, temperature=0.25)
            if not action_json or not isinstance(action_json, dict) or "action_type" not in action_json:
                error_msg = f"BŁĄD: Koder nie zwrócił poprawnej akcji JSON (próba {attempt + 1}). Otrzymano: {action_json}."
                print(error_msg)
                self._add_to_history(self.coder_history, "assistant", f"Invalid JSON response: {json.dumps(action_json)}")
                self._add_to_history(self.coder_history, "user", "Your previous response was not a valid JSON with 'action_type' or was missing required fields. Please analyze the system message for action types and their required formats, then try again. Ensure your entire response is a single JSON object.")
                if attempt + 1 == MAX_ITERATIONS_CODER:
                     return {"status": "SUBTASK_FAILED_INVALID_LLM_RESPONSE", "details": "Coder failed to provide valid JSON action after multiple attempts.", "history_excerpt": self.coder_history[-4:]}
                continue
            self._add_to_history(self.coder_history, "assistant", json.dumps(action_json))
            action_type = action_json.get("action_type")
            action_details = action_json.get("action_details", "")
            reasoning = action_json.get("reasoning", "No reasoning provided.")
            suggested_overall_timeout = action_json.get("suggested_overall_timeout")

            print(f"Koder wybrał akcję: {action_type}. Uzasadnienie: {reasoning}")
            if action_type != "ASK_HUMAN": print(f"Szczegóły akcji: '{action_details}'")
            if suggested_overall_timeout: print(f"Sugerowany timeout dla komendy: {suggested_overall_timeout}s")

            feedback_to_coder = ""
            current_error_this_turn_signature = None
            if action_type == "ASK_HUMAN":
                if isinstance(action_details, dict) and "question_for_human" in action_details:
                    question_for_human = action_details["question_for_human"]
                    print(f"INFO: Koder chce zadać pytanie użytkownikowi: {question_for_human}")
                    return {"status": "NEEDS_HUMAN_INPUT", "question": question_for_human, "reasoning": reasoning}
                else:
                    feedback_to_coder = "ERROR: Action 'ASK_HUMAN' was chosen, but 'action_details' was malformed. It must be a JSON object with a 'question_for_human' key. Example: {\"question_for_human\": \"What is the target API endpoint?\"}. Please try a different action or correct this one."
                    current_error_this_turn_signature = "ASK_HUMAN_MALFORMED"
            
            # ZMODYFIKOWANA SEKCJA PYTHON
            elif action_type == "PYTHON":
                if not isinstance(action_details, dict):
                    feedback_to_coder = (f"ERROR: 'action_details' for PYTHON action must be a JSON object. "
                                         f"Received type: {type(action_details)}. Expected format: "
                                         f"{{\"code\": \"...\", \"filename\": \"...\", \"execute\": true/false (optional)}}.")
                    current_error_this_turn_signature = "PYTHON_DETAILS_NOT_DICT"
                else:
                    code_string = action_details.get("code")
                    file_name = action_details.get("filename")
                    should_execute = action_details.get("execute", True)

                    validation_error = False
                    error_message_for_coder = ""

                    if not isinstance(code_string, str) or not code_string.strip():
                        error_message_for_coder = "ERROR: The 'code' field in 'action_details' for PYTHON action must be a non-empty string of Python code."
                        current_error_this_turn_signature = "PYTHON_INVALID_CODE_STRING"
                        validation_error = True
                    elif not isinstance(file_name, str) or not file_name.strip() or \
                         not file_name.endswith(".py") or \
                         file_name != os.path.basename(file_name) or \
                         not re.match(r"^[a-zA-Z0-9_.-]+$", file_name):
                        error_message_for_coder = (f"ERROR: The 'filename' ('{file_name}') in 'action_details' for PYTHON action "
                                                   f"must be a valid Python filename ending with '.py' (e.g., 'script.py'), "
                                                   f"contain only letters, numbers, underscores, hyphens, or periods, "
                                                   f"and must not include any path separators or be empty.")
                        current_error_this_turn_signature = "PYTHON_INVALID_FILENAME"
                        validation_error = True
                    elif not isinstance(should_execute, bool):
                        error_message_for_coder = "ERROR: The 'execute' field in 'action_details' for PYTHON action must be a boolean (true or false)."
                        current_error_this_turn_signature = "PYTHON_INVALID_EXECUTE_FLAG"
                        validation_error = True

                    if validation_error:
                        feedback_to_coder = error_message_for_coder
                    else:
                        stdout, stderr, r_code = execute_python_code(
                            code_string,
                            self.working_directory,
                            file_name,
                            execute_flag=should_execute
                        )
                        
                        action_summary = "written to file"
                        if should_execute:
                            action_summary = "written to file and execution attempted"

                        feedback_to_coder = (f"Python code for '{file_name}' was {action_summary}.\n"
                                             f"Return code (if executed): {r_code}\n"
                                             f"Stdout (if executed):\n{stdout}\n"
                                             f"Stderr (if executed):\n{stderr}\n")

                        if r_code != 0:
                            # Błąd jest istotny jeśli wykonywano, LUB jest to błąd krytyczny zapisu/walidacji z execute_python_code
                            if should_execute or r_code in [-110, -111]:
                                current_error_this_turn_signature = f"PYTHON_ERROR_CODE_{r_code}: {stderr[:150].strip()}" if stderr else f"PYTHON_ERROR_CODE_{r_code}_NO_STDERR"
                                if r_code == -110:
                                    current_error_this_turn_signature = f"PYTHON_PATH_SAFETY_ERROR_{r_code}: {stderr[:150].strip()}"
                                elif r_code == -111:
                                     current_error_this_turn_signature = f"PYTHON_EMPTY_FILENAME_ERROR_{r_code}: {stderr[:150].strip()}"
                                elif r_code < -100 and r_code not in [-110, -111]:
                                    current_error_this_turn_signature = f"PYTHON_INTERNAL_EXEC_ERROR_{r_code}: {stderr[:150].strip()}"
                                feedback_to_coder += "The script writing or execution FAILED. Analyze the error and try to fix it or choose a different action."
                        else: # r_code == 0
                            if should_execute:
                                feedback_to_coder += "The script executed successfully. What is the next step based on the sub-task requirements?"
                            else:
                                feedback_to_coder += "The script was written to file successfully (execution was skipped). What is the next step based on the sub-task requirements?"

            elif action_type == "TERMINAL":
                if not isinstance(action_details, str):
                    feedback_to_coder = f"ERROR: 'action_details' for TERMINAL action must be a command string. Received: {type(action_details)}. Please correct and try again."
                    current_error_this_turn_signature = "TERMINAL_INVALID_DETAILS"
                else:
                    overall_timeout_to_use = 300
                    if isinstance(suggested_overall_timeout, (int, float)) and suggested_overall_timeout > 0:
                        overall_timeout_to_use = int(suggested_overall_timeout)

                    stdout, stderr, r_code = execute_terminal_command(
                        action_details,
                        self.working_directory,
                        overall_timeout=overall_timeout_to_use
                    )
                    feedback_to_coder = f"Terminal command executed. Return code: {r_code}\nStdout:\n{stdout}\nStderr:\n{stderr}\n"
                    if r_code != 0:
                        current_error_this_turn_signature = f"TERMINAL_ERROR_CODE_{r_code}: {stderr[:150].strip()}" if stderr else f"TERMINAL_ERROR_CODE_{r_code}: Empty stderr"
                        if r_code <= -201 and r_code >= -205:
                             current_error_this_turn_signature = f"TERMINAL_EXEC_ERROR ({r_code}): {stderr[:150].strip()}"
                        feedback_to_coder += "The command FAILED. Analyze the error and try to fix it or choose a different action."
                    else: feedback_to_coder += "The command executed successfully. What is the next step based on the sub-task requirements?"
            elif action_type == "SEARCH":
                if not isinstance(action_details, str) or not action_details.strip():
                    feedback_to_coder = f"ERROR: 'action_details' for SEARCH action must be a non-empty search query string. Received: '{action_details}'. Please provide a valid query or choose a different action."
                    current_error_this_turn_signature = "SEARCH_INVALID_DETAILS"
                else:
                    print(f"Koder chce wyszukać: {action_details}")
                    search_results_list = search_duckduckgo(action_details, num_results=3)
                    formatted_search_results = "No results found for your query."
                    if search_results_list:
                        temp_researched_info_for_feedback = []
                        for res_idx, res in enumerate(search_results_list):
                            info_str = f"Search Result {res_idx+1}:\nTitle: {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}\nRelated_Query_Current_SubTask: '{action_details}'"
                            if not any(existing_info.startswith(f"Search Result {res_idx+1}:\nTitle: {res['title']}\nLink: {res['link']}") for existing_info in self.researched_info):
                                self.researched_info.append(info_str)
                            temp_researched_info_for_feedback.append(info_str)
                        formatted_search_results = "Search results (added to knowledge base):\n" + "\n".join(temp_researched_info_for_feedback)
                    feedback_to_coder = f"Search for '{action_details}' completed.\n{formatted_search_results}\nNow, what is your next action to progress on the sub-task: \"{current_sub_task_prompt}\"?"
            elif action_type == "LIST_FILES":
                path_to_list_str = action_details.strip() if isinstance(action_details, str) else ""
                target_path_to_list = self.working_directory
                additional_error_msg = ""

                if path_to_list_str:
                    normalized_path_to_list_str = path_to_list_str.lstrip('/\\')
                    potential_path = os.path.abspath(os.path.join(self.working_directory, normalized_path_to_list_str))

                    if potential_path.startswith(self.working_directory) and os.path.isdir(potential_path):
                        target_path_to_list = potential_path
                    elif not os.path.exists(potential_path):
                         additional_error_msg = f"ERROR: Requested path '{path_to_list_str}' for LIST_FILES does not exist. Listing project root '{self.working_directory}' instead.\n"
                         target_path_to_list = self.working_directory
                    elif not os.path.isdir(potential_path):
                         additional_error_msg = f"ERROR: Requested path '{path_to_list_str}' for LIST_FILES is not a directory. Listing project root '{self.working_directory}' instead.\n"
                         target_path_to_list = self.working_directory
                    else:
                        additional_error_msg = f"ERROR: Requested path '{path_to_list_str}' for LIST_FILES is outside the allowed working directory '{self.working_directory}'. Listing project root instead.\n"
                        target_path_to_list = self.working_directory

                actual_display_rel_path = os.path.relpath(target_path_to_list, self.working_directory)
                if actual_display_rel_path == ".": actual_display_rel_path = ""

                feedback_to_coder = additional_error_msg
                try:
                    if os.path.exists(target_path_to_list) and os.path.isdir(target_path_to_list):
                        items = os.listdir(target_path_to_list)
                        dir_name_for_output = f"'{actual_display_rel_path}' (relative to project root)" if actual_display_rel_path else f"project root ('{self.working_directory}')"
                        files_list_output = f"Contents of directory {dir_name_for_output}:\n"

                        if not items: files_list_output += "- (empty)\n"
                        else:
                            for item in sorted(items):
                                item_full_path = os.path.join(target_path_to_list, item)
                                item_type = "DIR" if os.path.isdir(item_full_path) else "FILE"
                                files_list_output += f"- {item} ({item_type})\n"
                        feedback_to_coder += files_list_output
                    else:
                        dir_name_for_error = f"'{actual_display_rel_path}'" if actual_display_rel_path else f"project root ('{self.working_directory}')"
                        feedback_to_coder += f"ERROR: Path {dir_name_for_error} does not exist or is not a directory (checked again before listing).\n"
                except Exception as e:
                    dir_name_for_error = f"'{actual_display_rel_path}'" if actual_display_rel_path else f"project root ('{self.working_directory}')"
                    feedback_to_coder += f"ERROR listing files in {dir_name_for_error}: {e}\n"

                feedback_to_coder += f"Now, what is your next action to progress on the sub-task: \"{current_sub_task_prompt}\"?"

            elif action_type == "FINISHED_SUBTASK":
                print(f"INFO: Koder zgłasza ukończenie podzadania: {action_details}")
                return {"status": "SUBTASK_COMPLETED", "details": action_details if isinstance(action_details, str) else "Sub-task finished.", "history_excerpt": self.coder_history[-2:]}
            elif action_type == "CANNOT_COMPLETE_SUBTASK":
                print(f"INFO: Koder zgłasza niemożność ukończenia podzadania: {action_details}")
                return {"status": "SUBTASK_FAILED", "details": action_details if isinstance(action_details, str) else "Cannot complete sub-task.", "history_excerpt": self.coder_history[-2:]}
            else:
                feedback_to_coder = f"ERROR: Unknown or malformed action_type: '{action_type}'. Please choose from the valid action types listed in the system message and ensure correct 'action_details' format. Specifically, check if 'action_details' is required and its type for the chosen action."
                current_error_this_turn_signature = f"UNKNOWN_ACTION: {action_type}"

            if current_error_this_turn_signature:
                if current_error_this_turn_signature == previous_error_signature:
                    consecutive_error_count += 1
                else:
                    previous_error_signature = current_error_this_turn_signature
                    consecutive_error_count = 1

                if consecutive_error_count >= MAX_CONSECUTIVE_ERRORS_CODER:
                    print(f"KRYTYCZNY BŁĄD KODERA: Ten sam błąd ('{previous_error_signature}') wystąpił {consecutive_error_count} razy z rzędu.")
                    return {"status": "SUBTASK_REPEATED_ERRORS",
                            "details": f"Coder failed after {consecutive_error_count} consecutive identical errors: '{previous_error_signature}'. Last action: {action_type} - '{action_details}'",
                            "history_excerpt": self.coder_history[-2:]}
            else:
                previous_error_signature = None
                consecutive_error_count = 0

            print(f"Feedback dla Kodera: {feedback_to_coder}")
            if feedback_to_coder: self._add_to_history(self.coder_history, "user", feedback_to_coder)
            time.sleep(0.2)

        print(f"INFO: Koder osiągnął maksymalną liczbę iteracji ({MAX_ITERATIONS_CODER}) dla tego podzadania.")
        return {"status": "SUBTASK_MAX_ITERATIONS", "details": f"Coder reached max iterations ({MAX_ITERATIONS_CODER}) for sub-task: \"{current_sub_task_prompt}\".", "history_excerpt": self.coder_history[-2:]}

    def run(self):
        if not self.planner_phase():
            print(f"KRYTYCZNY BŁĄD: Faza planowania nie powiodła się dla zadania \"{self.initial_task}\". Zakończenie tego cyklu zadania.")
            return {"status": "PLANNER_INIT_FAILED", "details": f"Initial planning and research phase failed for task: {self.initial_task}"}

        system_msg_planner_evaluator_template = """
You are an AI Project Manager. Your role is to oversee the Coder AI's progress, evaluate its work on sub-tasks, and guide it towards completing the overall project goal for THE CURRENT TASK.
The Coder operates on a {platform_system} system in the working directory: "{working_directory}". This directory may contain existing files/venv from prior tasks.
The venv name is 'my_agent_venv'.

Overall Project Goal (for the current task, clarified with user): "{initial_task}"
Initial High-Level Plan (generated by Planner for THIS task): {overall_plan_json}
Relevant Information Gathered During Research (includes all research in this session, use your judgment for relevance to current task): {researched_info_json}

Current state of the plan for THIS task:
- Completed plan steps so far (for THIS task): {completed_plan_steps_json}
- Remaining plan steps (for THIS task): {remaining_plan_steps_json}

You will receive updates from the Coder AI after it attempts a sub-task.
The Coder's update will include its 'status' for the sub-task and 'details'. Possible statuses:
- SUBTASK_COMPLETED: The Coder believes it successfully finished the assigned sub-task.
- SUBTASK_FAILED: The Coder was unable to complete the sub-task due to a specific issue.
- SUBTASK_MAX_ITERATIONS: The Coder reached its iteration limit for the sub-task without success.
- SUBTASK_REPEATED_ERRORS: The Coder encountered the same error multiple times and is stuck.
- SUBTASK_FAILED_INVALID_LLM_RESPONSE: The Coder had issues generating valid actions.

Your responsibilities:
1. Evaluate the Coder's status and details in the context of the current sub-task and the overall plan FOR THIS TASK.
2. If 'SUBTASK_COMPLETED':
   - Assess if the Coder's reported completion aligns with the sub-task's objective.
   - If yes, and there are more steps in THIS task's plan, determine the next logical sub-task.
   - If yes, and this was the last step OF THIS TASK's plan, decide 'TASK_COMPLETED_SUCCESSFULLY'.
3. If the Coder failed (SUBTASK_FAILED, MAX_ITERATIONS, REPEATED_ERRORS, INVALID_LLM_RESPONSE):
   - Analyze the Coder's reasoning and history (provided in the update details/history_excerpt).
   - Decide on a course of action:
     - Provide a corrective instruction or a rephrased/simplified version of the SAME sub-task if you believe the Coder can succeed with guidance.
     - Assign a DIFFERENT sub-task from THIS TASK's plan if the failed one is too problematic for now, or if an alternative approach is better.
     - If the failure is critical or THIS TASK's plan seems unachievable, decide 'TASK_FAILED_CANNOT_COMPLETE'.
   - Do NOT simply ask the Coder to "try again" without specific new guidance if it reported 'MAX_ITERATIONS' or 'REPEATED_ERRORS'. It needs a change in approach.
4. The Coder might have interacted with the human user via 'ASK_HUMAN'. Consider this interaction.

Respond in JSON format with "decision", "next_coder_task", "reasoning".
- "decision": One of ["CONTINUE_CODING", "TASK_COMPLETED_SUCCESSFULLY", "TASK_FAILED_CANNOT_COMPLETE"]
- "next_coder_task": A string. If "CONTINUE_CODING", this is the specific, actionable instruction for the Coder's next sub-task for THIS task. If other decisions, this can be empty or a summary of THIS task's outcome.
- "reasoning": Your detailed thought process for the decision and the next task.

Ensure your response is a single, valid JSON object.
"""
        current_task_for_coder = ""
        if self.overall_plan:
            current_task_for_coder = self.overall_plan[0]
        else:
            print(f"OSTRZEŻENIE KRYTYCZNE: Brak `overall_plan` po fazie planowania dla zadania \"{self.initial_task}\". Używam `initial_task` jako pierwszego zadania dla Kodera.")
            current_task_for_coder = f"The overall goal for this task is: {self.initial_task}. Since no detailed plan was available, please start working on this main goal. Break it down if necessary as you proceed."
            self.overall_plan = [current_task_for_coder]

        completed_plan_steps_list = []
        current_plan_step_index = 0
        final_result = {}
        planner_iteration_count = 0

        while True:
            planner_iteration_count +=1
            print(f"\n--- Pętla Planisty-Oceniającego: Iteracja {planner_iteration_count} (dla zadania: \"{self.initial_task}\") ---")
            print(f"Ogólny cel bieżącego zadania: \"{self.initial_task}\"")
            print("Aktualny plan dla bieżącego zadania:")
            for i, step in enumerate(self.overall_plan):
                status = "✅ Ukończony" if step in completed_plan_steps_list else ("⏳ Bieżący/Następny" if i == current_plan_step_index else "🔲 Oczekujący")
                print(f"  {i+1}. {step} [{status}]")

            if not self.resuming_coder_after_human_input:
                self.coder_history = []
                initial_coder_prompt = (
                    f"Your current sub-task, as assigned by the Project Manager for the overall task \"{self.initial_task}\", is: \"{current_task_for_coder}\".\n"
                    f"Please review the overall project goal for this task, the complete plan for this task, and any relevant research information provided in the system message (which may include info from previous project tasks).\n"
                    f"Focus on achieving this specific sub-task. What is your first action?"
                )
                self._add_to_history(self.coder_history, "user", initial_coder_prompt)

            self.resuming_coder_after_human_input = False

            coder_result = self.coder_phase_loop(current_task_for_coder)

            if coder_result.get("status") == "NEEDS_HUMAN_INPUT":
                print(f"\n🤖 AGENT POTRZEBUJE TWOJEJ POMOCY (dla zadania: \"{self.initial_task}\") 🤖")
                question_from_coder = coder_result.get("question", "Coder needs input, but no question was provided.")
                coder_reasoning_for_question = coder_result.get("reasoning", "No reasoning provided for asking.")
                print(f"Koder pyta (uzasadnienie: {coder_reasoning_for_question}):\n   \"{question_from_coder}\"")

                human_answer = ""
                while not human_answer.strip():
                    human_answer = input("Twoja odpowiedź: > ").strip()
                    if not human_answer: print("Odpowiedź nie może być pusta. Spróbuj ponownie.")

                feedback_after_human_input = (
                    f"The Coder previously asked the human user: '{question_from_coder}' (Reasoning: {coder_reasoning_for_question}).\n"
                    f"Human's answer: '{human_answer}'\n"
                    f"Based on this new information, please continue with your sub-task: \"{current_task_for_coder}\" (which is part of overall task \"{self.initial_task}\"). What is your next action?"
                )
                self._add_to_history(self.coder_history, "user", feedback_after_human_input)
                self.resuming_coder_after_human_input = True
                continue

            coder_update_str = f"Coder's status for sub-task \"{current_task_for_coder}\" (part of overall task \"{self.initial_task}\"): {coder_result['status']}.\nDetails/Summary: {coder_result.get('details', 'N/A')}.\n"
            if "history_excerpt" in coder_result and coder_result["history_excerpt"]:
                coder_update_str += f"Excerpt from Coder's recent internal history for this sub-task:\n{json.dumps(coder_result['history_excerpt'], indent=2)}\n"

            current_completed_for_prompt = list(completed_plan_steps_list)
            remaining_for_prompt = []

            if self.overall_plan:
                if coder_result["status"] == "SUBTASK_COMPLETED" and \
                   current_plan_step_index < len(self.overall_plan) and \
                   self.overall_plan[current_plan_step_index] == current_task_for_coder:

                    temp_completed_for_prompt = list(current_completed_for_prompt)
                    if self.overall_plan[current_plan_step_index] not in temp_completed_for_prompt:
                         temp_completed_for_prompt.append(self.overall_plan[current_plan_step_index])

                    if current_plan_step_index + 1 < len(self.overall_plan):
                        remaining_for_prompt = self.overall_plan[current_plan_step_index+1:]
                    current_completed_for_prompt = temp_completed_for_prompt
                else:
                    if current_plan_step_index < len(self.overall_plan):
                         remaining_for_prompt = self.overall_plan[current_plan_step_index:]

            system_msg_planner_evaluator = system_msg_planner_evaluator_template.format(
                platform_system=self.platform_system, working_directory=self.working_directory,
                initial_task=self.initial_task,
                overall_plan_json=json.dumps(self.overall_plan),
                researched_info_json=json.dumps(self.researched_info[-5:] if self.researched_info else "No research performed or info gathered."),
                completed_plan_steps_json=json.dumps(current_completed_for_prompt),
                remaining_plan_steps_json=json.dumps(remaining_for_prompt)
            )

            planner_eval_messages = [{"role":"system", "content":system_msg_planner_evaluator}]
            planner_eval_messages.append({"role": "user", "content": f"Update from Coder:\n{coder_update_str}\n\nBased on this update, the current plan status for task \"{self.initial_task}\", and this task's goal, what is your decision and the next sub-task for the Coder (if any)? Please provide your response as a JSON object."})

            self._add_to_history(self.planner_history, "user", planner_eval_messages[-1]['content'])

            planner_decision_json = ask_llm(messages_override=planner_eval_messages, model=MODEL_PLANNER, json_mode=True, temperature=0.35)

            if not planner_decision_json:
                print(f"BŁĄD KRYTYCZNY: Planista-Oceniający nie zwrócił żadnej odpowiedzi (None) dla zadania \"{self.initial_task}\". Zakończenie tego cyklu zadania.")
                final_result = {"status": "PLANNER_LLM_NO_RESPONSE", "coder_last_status": coder_result, "details": f"Planner-Evaluator LLM returned None for task: {self.initial_task}."}
                self._add_to_history(self.planner_history, "assistant", "LLM returned None for planner decision.")
                break

            if not isinstance(planner_decision_json, dict) or "decision" not in planner_decision_json or "next_coder_task" not in planner_decision_json :
                print(f"BŁĄD: Planista-Oceniający nie zwrócił poprawnej decyzji JSON (brak 'decision' lub 'next_coder_task') dla zadania \"{self.initial_task}\". Otrzymano: {planner_decision_json}. Zakończenie tego cyklu zadania.")
                final_result = {"status": "PLANNER_ERROR_INVALID_JSON", "coder_last_status": coder_result, "details": f"Planner-Evaluator failed to provide valid JSON decision for task {self.initial_task}. Got: {planner_decision_json}"}
                self._add_to_history(self.planner_history, "assistant", f"Invalid JSON response: {json.dumps(planner_decision_json)}")
                break

            self._add_to_history(self.planner_history, "assistant", json.dumps(planner_decision_json))

            decision = planner_decision_json.get("decision")
            next_task_from_planner = planner_decision_json.get("next_coder_task", "").strip()
            reasoning = planner_decision_json.get("reasoning", "No reasoning provided.")

            print(f"Decyzja Planisty-Oceniającego (dla zadania \"{self.initial_task}\"): {decision}. Uzasadnienie: {reasoning}")
            if decision == "CONTINUE_CODING": print(f"Następne zadanie dla Kodera: \"{next_task_from_planner}\"")

            if decision == "TASK_COMPLETED_SUCCESSFULLY":
                if coder_result["status"] == "SUBTASK_COMPLETED":
                    print(f"\n🏆 BIEŻĄCE ZADANIE UKOŃCZONE POMYŚLNIE! ({self.initial_task}) 🏆")
                    final_result = {"status": "SUCCESS", "details": reasoning, "coder_last_result_details": coder_result.get("details")}
                    if current_plan_step_index < len(self.overall_plan) and \
                       self.overall_plan[current_plan_step_index] not in completed_plan_steps_list:
                        completed_plan_steps_list.append(self.overall_plan[current_plan_step_index])
                    break
                else:
                    print(f"OSTRZEŻENIE: Planista-Oceniający zgłosił TASK_COMPLETED_SUCCESSFULLY dla zadania \"{self.initial_task}\", ale ostatnie zadanie Kodera ({coder_result['status']}) nie zakończyło się statusem SUBTASK_COMPLETED. Zmieniam decyzję na TASK_FAILED_CANNOT_COMPLETE z powodu niespójności.")
                    decision = "TASK_FAILED_CANNOT_COMPLETE"
                    reasoning += f" (Auto-changed for task '{self.initial_task}': Inconsistency between Planner's 'SUCCESS' and Coder's last sub-task status being not 'SUBTASK_COMPLETED')"

            if decision == "TASK_FAILED_CANNOT_COMPLETE":
                print(f"\n❌ BIEŻĄCE ZADANIE NIEUDANE - NIE MOŻNA UKOŃCZYĆ. ({self.initial_task}) ❌")
                final_result = {"status": "FAILURE", "details": reasoning, "coder_last_result_details": coder_result.get("details")}
                break

            elif decision == "CONTINUE_CODING":
                if not next_task_from_planner:
                    print(f"KRYTYCZNE OSTRZEŻENIE: Planista-Oceniający zdecydował 'CONTINUE_CODING' dla zadania \"{self.initial_task}\", ale nie podał 'next_coder_task'. To jest błąd w logice Planisty-Oceniającego. Traktuję to jako TASK_FAILED.")
                    final_result = {"status": "FAILURE", "details": f"Planner-Evaluator decided to continue for task '{self.initial_task}' but provided no next task.", "coder_last_result_details": coder_result.get("details")}
                    break

                if coder_result["status"] == "SUBTASK_COMPLETED" and \
                   current_plan_step_index < len(self.overall_plan) and \
                   self.overall_plan[current_plan_step_index] == current_task_for_coder:

                    if self.overall_plan[current_plan_step_index] not in completed_plan_steps_list:
                         completed_plan_steps_list.append(self.overall_plan[current_plan_step_index])
                    current_plan_step_index += 1

                current_task_for_coder = next_task_from_planner

            else:
                print(f"OSTRZEŻENIE KRYTYCZNE: Nieznana lub nieobsługiwana decyzja Planisty-Oceniającego: {decision} (dla zadania \"{self.initial_task}\"). Zakończenie z błędem.")
                final_result = {"status": "UNKNOWN_PLANNER_DECISION", "details": f"Unknown decision for task '{self.initial_task}': {decision}", "reasoning": reasoning}
                break

            time.sleep(0.2)

        print(f"\n--- Zakończenie cyklu pracy agenta dla zadania \"{self.initial_task}\" (wewnątrz dedykowanego folderu projektu) ---")
        history_dir = os.path.join(self.working_directory, "agent_history")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_task_name_part = re.sub(r'\W+', '_', self.initial_task[:30])

        planner_file = os.path.join(history_dir, f"planner_evaluator_history_{safe_task_name_part}_{timestamp}.json")
        coder_last_subtask_file = os.path.join(history_dir, f"coder_last_subtask_history_{safe_task_name_part}_{timestamp}.json")

        try:
            with open(planner_file, "w", encoding="utf-8") as f: json.dump(self.planner_history, f, indent=2, ensure_ascii=False)
            if self.coder_history:
                 with open(coder_last_subtask_file, "w", encoding="utf-8") as f: json.dump(self.coder_history, f, indent=2, ensure_ascii=False)
            print(f"Historie dla zadania \"{self.initial_task}\" zapisane w katalogu: {history_dir}")
        except Exception as e:
            print(f"OSTRZEŻENIE: Nie udało się zapisać plików historii dla zadania \"{self.initial_task}\": {e}")

        print(f"Ostateczny status dla zadania \"{self.initial_task}\": {final_result.get('status')}")
        print(f"Szczegóły: {final_result.get('details')}")
        if final_result.get('coder_last_result_details'):
            print(f"Szczegóły ostatniego wyniku Kodera (sub-zadanie): {final_result.get('coder_last_result_details')}")

        return final_result

# --- Uruchomienie ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("BŁĄD: Skrypt wymaga dwóch argumentów: <ścieżka_do_folderu_projektu> \"<zadanie_użytkownika>\"")
        print("Przykład Użycia (dla Linux/macOS, dostosuj ścieżkę dla Windows):")
        print(f"  python {os.path.basename(__file__)} \"/Users/twojanazwa/fromai/MojProjekt_YYYYMMDD_HHMMSS\" \"Stwórz prostą stronę HTML\"")
        print("Pamiętaj, że folder projektu powinien być utworzony przez skrypt launchera (np. start_agent.sh lub ręcznie).")
        sys.exit(2)

    project_folder_path_from_cli = os.path.abspath(sys.argv[1])
    raw_user_task_from_cli = sys.argv[2]

    print(f"INFO: Otrzymano ścieżkę roboczą projektu z CLI: {project_folder_path_from_cli}")
    print(f"INFO: Otrzymano surowe zadanie użytkownika z CLI: \"{raw_user_task_from_cli}\"")

    if not os.path.isdir(project_folder_path_from_cli):
        try:
            print(f"INFO: Folder projektu '{project_folder_path_from_cli}' nie istnieje. Próba utworzenia...")
            os.makedirs(project_folder_path_from_cli, exist_ok=True)
            print(f"INFO: Pomyślnie utworzono folder projektu: '{project_folder_path_from_cli}'")
        except OSError as e:
            print(f"KRYTYCZNY BŁĄD: Folder projektu przekazany z CLI nie istnieje ('{project_folder_path_from_cli}') i nie można go utworzyć: {e}")
            sys.exit(1)

    current_project_folder_path = project_folder_path_from_cli
    current_user_task = raw_user_task_from_cli
    
    agent = None
    is_first_task_in_session = True 

    while True:
        if is_first_task_in_session:
            print("\n--- Rozpoczynanie fazy doprecyzowania zadania (pierwsze zadanie w sesji) ---")
            clarified_task = clarify_initial_task_with_user(current_user_task, current_project_folder_path, is_continuation=False)
        else:
            print(f"\n--- Doprecyzowanie nowego zadania dla projektu w '{current_project_folder_path}' ---")
            clarified_task = clarify_initial_task_with_user(current_user_task, current_project_folder_path, is_continuation=True)

        if not clarified_task:
            print("Nie udało się uzyskać doprecyzowanego zadania. Zakończenie pracy nad projektem.")
            sys.exit(1)
        
        print(f"\nRozpoczynanie pracy głównego agenta z zadaniem: \"{clarified_task}\"")
        if agent is None:
            agent = AIAgent(initial_task=clarified_task, project_working_directory=current_project_folder_path)
        else:
            agent.reset_for_new_task(clarified_task)
        
        is_first_task_in_session = False
        result = agent.run()

        print(f"\n--- Zakończono cykl pracy agenta XD.py dla zadania: \"{clarified_task}\" ---")
        if result:
            print(f"Wynik cyklu pracy agenta: {result.get('status')} - {result.get('details')}")

        if result and result.get('status') == "SUCCESS":
            print(f"🏆 Zadanie \"{clarified_task}\" ukończone pomyślnie! 🏆")
            while True:
                user_choice = input(f"Czy chcesz kontynuować rozwój projektu w '{current_project_folder_path}' (np. dodać nowe funkcje, zmodyfikować istniejące)? (tak/nie): ").strip().lower()
                if user_choice in ["tak", "t", "yes", "y"]:
                    new_task_prompt = ""
                    while not new_task_prompt:
                        new_task_prompt = input("Opisz, co chciałbyś teraz zrobić z tym projektem: ").strip()
                        if not new_task_prompt:
                            print("Opis zadania nie może być pusty.")
                    current_user_task = new_task_prompt
                    break 
                elif user_choice in ["nie", "n", "no"]:
                    print(f"Zakończono pracę nad projektem w '{current_project_folder_path}'. Do widzenia!")
                    sys.exit(0)
                else:
                    print("Nierozpoznana odpowiedź. Wpisz 'tak' lub 'nie'.")
        else:
            print(f"Zadanie \"{clarified_task}\" nie zostało ukończone pomyślnie. Zakończenie pracy nad projektem w '{current_project_folder_path}'.")
            sys.exit(1)
