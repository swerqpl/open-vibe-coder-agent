#!/bin/bash

PYTHON_EXEC="python3.10" # Dostosuj w razie potrzeby
AGENT_RUNNER_VENV_NAME="agent_bootstrap_venv" # venv dla samego launchera

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Katalog gdzie jest launcher i XD.py
AGENT_MAIN_SCRIPT_PATH="$SCRIPT_DIR/XD.py"
VENV_LAUNCHER_PATH="$SCRIPT_DIR/$AGENT_RUNNER_VENV_NAME"

# --- Konfiguracja głównego folderu dla wszystkich projektów AI ---
BASE_PROJECTS_DIR="/Users/rafalkorszun/fromai" # Twój docelowy folder

# Zmienna przechowująca ścieżkę do folderu BIEŻĄCEGO projektu
# Zostanie ustawiona po uzyskaniu nazwy zadania od użytkownika
CURRENT_PROJECT_FOLDER_PATH=""

# --- Funkcja do czyszczenia przy wyjściu/przerwaniu ---
cleanup() {
    echo ""
    echo "INFO: Otrzymano sygnał przerwania lub zakończenia. Rozpoczynanie czyszczenia..."
    if command -v deactivate &> /dev/null && [[ "$(type -t deactivate)" == "function" ]]; then
        echo "INFO: Deaktywacja środowiska wirtualnego '$AGENT_RUNNER_VENV_NAME'..."
        deactivate
    fi

    # Czyść tylko jeśli CURRENT_PROJECT_FOLDER_PATH jest ustawione i folder istnieje
    if [ -n "$CURRENT_PROJECT_FOLDER_PATH" ] && [ -d "$CURRENT_PROJECT_FOLDER_PATH" ]; then
        if [ "$INTERRUPTED_BY_USER" = "true" ]; then
            read -r -p "Czy chcesz usunąć folder bieżącego projektu '$CURRENT_PROJECT_FOLDER_PATH'? [t/N] " response
            if [[ "$response" =~ ^([tT])$ ]]; then
                echo "INFO: Usuwanie folderu projektu '$CURRENT_PROJECT_FOLDER_PATH'..."
                rm -rf "$CURRENT_PROJECT_FOLDER_PATH"
                if [ $? -eq 0 ]; then echo "INFO: Folder projektu usunięty."; else echo "OSTRZEŻENIE: Nie udało się usunąć folderu projektu."; fi
            else
                echo "INFO: Folder projektu NIE został usunięty."
            fi
        else
             # Jeśli nie przerwano, a np. XD.py zwrócił błąd, nie usuwamy automatycznie
            echo "INFO: Skrypt zakończony. Folder projektu '$CURRENT_PROJECT_FOLDER_PATH' NIE został automatycznie usunięty."
        fi
    elif [ "$INTERRUPTED_BY_USER" = "true" ]; then
        echo "INFO: Brak aktywnego folderu projektu do usunięcia lub folder nie istnieje."
    fi
    echo "INFO: Czyszczenie zakończone."
}

INTERRUPTED_BY_USER="false"
trap 'INTERRUPTED_BY_USER="true"; cleanup; exit 130' INT
trap 'cleanup' TERM
# EXIT trap zostanie obsłużony na końcu

# --- Główna logika skryptu ---
echo "INFO: Rozpoczynanie pracy launchera agenta..."

# 1. Utwórz główny folder fromai, jeśli nie istnieje
mkdir -p "$BASE_PROJECTS_DIR"
if [ $? -ne 0 ]; then
    echo "BŁĄD: Nie udało się utworzyć głównego folderu projektów '$BASE_PROJECTS_DIR'."
    exit 1
fi
echo "INFO: Główny folder dla wszystkich projektów AI: $BASE_PROJECTS_DIR"

# 2. Sprawdzenie Pythona i utworzenie/aktywacja venv dla launchera (w katalogu skryptu)
if ! command -v $PYTHON_EXEC &> /dev/null; then # POPRAWIONY BLOK
    echo "BŁĄD: Nie znaleziono interpretera Python $PYTHON_EXEC."
    exit 1
fi
echo "INFO: Używany interpreter Python do tworzenia venv launchera: $($PYTHON_EXEC --version)"

if [ ! -d "$VENV_LAUNCHER_PATH" ]; then # POPRAWIONY BLOK
    echo "INFO: Tworzenie środowiska wirtualnego launchera '$AGENT_RUNNER_VENV_NAME' w '$SCRIPT_DIR'..."
    $PYTHON_EXEC -m venv "$VENV_LAUNCHER_PATH" || { echo "BŁĄD: Nie udało się utworzyć venv launchera."; exit 1; }
else
    echo "INFO: Środowisko wirtualne launchera '$AGENT_RUNNER_VENV_NAME' już istnieje."
fi

# shellcheck disable=SC1091
source "$VENV_LAUNCHER_PATH/bin/activate" || { echo "BŁĄD: Nie udało się aktywować venv launchera."; exit 1; }
echo "INFO: Aktywowano venv launchera. Używany Python: $(python --version)"

pip install --upgrade pip > /dev/null 2>&1
pip install -qq openai requests beautifulsoup4 python-dotenv # Dodaj 'trafilatura' jeśli używasz
if [ $? -ne 0 ]; then # POPRAWIONY BLOK
    echo "BŁĄD: Nie udało się zainstalować pakietów dla agenta."
    deactivate; exit 1;
fi
echo "INFO: Pakiety dla agenta zainstalowane/zaktualizowane."


# 3. Uzyskaj zadanie od użytkownika (Launcher pyta, bo potrzebujemy tego do nazwy folderu)
echo "----------------------------------------------------------------------"
echo "--- Konfiguracja Agenta ---"
read -r -p "Podaj zadanie dla agenta AI (np. 'stwórz plik test.txt z tekstem Hello'): " RAW_USER_TASK
if [ -z "$RAW_USER_TASK" ]; then
    echo "Nie podano zadania. Zakończenie."
    deactivate; exit 1;
fi
echo "----------------------------------------------------------------------"

# 4. Wygeneruj nazwę folderu dla bieżącego projektu
PROJECT_NAME_BASE=$(echo "$RAW_USER_TASK" | awk '{for(i=1;i<=5 && i<=NF;i++) printf $i"_"}')
PROJECT_NAME_BASE=$(echo "${PROJECT_NAME_BASE%_}" | sed 's/[^a-zA-Z0-9_]/_/g' | cut -c 1-50)
if [ -z "$PROJECT_NAME_BASE" ]; then
    PROJECT_NAME_BASE="Untitled_Project"
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CURRENT_PROJECT_FOLDER_NAME="${PROJECT_NAME_BASE}_${TIMESTAMP}"
CURRENT_PROJECT_FOLDER_PATH="$BASE_PROJECTS_DIR/$CURRENT_PROJECT_FOLDER_NAME"

echo "INFO: Folder dla bieżącego projektu zostanie utworzony w: $CURRENT_PROJECT_FOLDER_PATH"

# 5. Utwórz unikalny folder projektu
mkdir -p "$CURRENT_PROJECT_FOLDER_PATH"
if [ $? -ne 0 ]; then
    echo "BŁĄD: Nie udało się utworzyć folderu dla bieżącego projektu '$CURRENT_PROJECT_FOLDER_PATH'."
    deactivate; exit 1;
fi
echo "INFO: Utworzono folder dla bieżącego projektu."

# 6. Uruchom główny skrypt agenta (XD.py), przekazując mu RAW_USER_TASK i CURRENT_PROJECT_FOLDER_PATH
echo "----------------------------------------------------------------------"
echo "INFO: Uruchamianie agenta XD.py..."
echo "      Zadanie: $RAW_USER_TASK"
echo "      Folder roboczy: $CURRENT_PROJECT_FOLDER_PATH"
echo "----------------------------------------------------------------------"
python "$AGENT_MAIN_SCRIPT_PATH" "$CURRENT_PROJECT_FOLDER_PATH" "$RAW_USER_TASK"
AGENT_EXIT_CODE=$?
echo "----------------------------------------------------------------------"
echo "INFO: Agent XD.py zakończył działanie z kodem: $AGENT_EXIT_CODE"
echo "----------------------------------------------------------------------"

# 7. Logika po zakończeniu XD.py (np. tylko informacyjna, bo folder już ma właściwą nazwę)
if [ $AGENT_EXIT_CODE -eq 0 ]; then
    echo "INFO: Agent zakończył pracę pomyślnie. Projekt znajduje się w: $CURRENT_PROJECT_FOLDER_PATH"
else
    echo "OSTRZEŻENIE: Agent XD.py zakończył się z błędem (kod: $AGENT_EXIT_CODE). Sprawdź logi w $CURRENT_PROJECT_FOLDER_PATH."
fi

# Deaktywacja i końcowe czyszczenie (jeśli przerwano)
INTERRUPTED_BY_USER="false" # Resetuj, bo doszliśmy do normalnego końca
cleanup
echo "INFO: Skrypt launchera zakończony z kodem $AGENT_EXIT_CODE."
exit $AGENT_EXIT_CODE
