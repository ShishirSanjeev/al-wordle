import sys
import math
import random
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QGridLayout, QPushButton, QTextEdit, QInputDialog, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QEventLoop

from PySide6.QtGui import QTextCursor

# Optional: Use numba for JIT compilation if available
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# --- Efficient feedback pattern encoding ---
@jit(nopython=True, cache=True)
def feedback_code_numba(guess, answer):
    result = np.zeros(5, dtype=np.int32)
    used = np.zeros(5, dtype=np.bool_)

    # Green pass
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 2
            used[i] = True

    # Yellow pass
    for i in range(5):
        if result[i] == 2:
            continue
        for j in range(5):
            if not used[j] and guess[i] == answer[j]:
                result[i] = 1
                used[j] = True
                break

    # Encode as base-3 integer
    code = 0
    for i in range(5):
        code = code * 3 + result[i]
    return code

@lru_cache(maxsize=None)
def feedback_code(guess, answer):
    if NUMBA_AVAILABLE:
        return feedback_code_numba(guess, answer)

    result = [0] * 5
    used = [False] * 5
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 2  # G
            used[i] = True
    for i in range(5):
        if result[i] == 2:
            continue
        for j in range(5):
            if not used[j] and guess[i] == answer[j]:
                result[i] = 1  # Y
                used[j] = True
                break
    # Encode as a base-3 integer
    code = 0
    for v in result:
        code = code * 3 + v
    return code

# --- Vectorized feedback calculation with caching ---
_feedback_cache = {}
def feedback_matrix(guess, candidates):
    cache_key = (guess, tuple(candidates))
    if cache_key in _feedback_cache:
        return _feedback_cache[cache_key]

    codes = np.array([feedback_code(guess, word) for word in candidates], dtype=np.int32)
    _feedback_cache[cache_key] = codes
    return codes

# --- Early stopping in entropy calculation ---
def expected_entropy_early(guess, candidates, word_probs, entropy_threshold=None):
    codes = feedback_matrix(guess, candidates)

    # Pre-compute probabilities array for candidates
    if isinstance(word_probs, dict):
        probs = np.array([word_probs.get(word, 1/len(candidates)) for word in candidates], dtype=np.float64)
    else:
        probs = word_probs

    max_code = codes.max() + 1
    group_probs = np.bincount(codes, weights=probs, minlength=max_code)

    # Vectorized entropy calculation
    mask = group_probs > 0
    if not mask.any():
        return 0.0

    total_entropy = np.sum(group_probs[mask] * np.log2(1 / group_probs[mask]))

    # Early stopping
    if entropy_threshold is not None and total_entropy > entropy_threshold:
        return total_entropy
    return total_entropy

# Worker thread for entropy calculation with progress and cancellation
class EntropyWorker(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    cancelled = Signal()

    def __init__(self, guess_pool, candidates, word_probs, entropy_threshold=None):
        super().__init__()
        self.guess_pool = guess_pool
        self.candidates = candidates
        self.word_probs = word_probs
        self.entropy_threshold = entropy_threshold
        self._cancel = False

    def run(self):
        entropies = {}
        total = len(self.guess_pool)

        # Use ProcessPoolExecutor for parallel computation
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:
            # Submit all tasks
            future_to_guess = {
                executor.submit(expected_entropy_early, word, self.candidates, self.word_probs, self.entropy_threshold): word
                for word in self.guess_pool
            }

            completed = 0
            for future in as_completed(future_to_guess):
                if self._cancel:
                    executor.shutdown(wait=False)
                    self.cancelled.emit()
                    return

                word = future_to_guess[future]
                try:
                    entropies[word] = future.result()
                except Exception as e:
                    print(f"Error calculating entropy for {word}: {e}")
                    entropies[word] = float('inf')

                completed += 1
                if completed % max(1, total // 20) == 0 or completed == total:
                    self.progress.emit(int((completed / total) * 100))

        self.finished.emit(entropies)

    def cancel(self):
        self._cancel = True

# Load answer and guess word lists from files
def load_words(answer_file='answers.txt', guess_file='guesses.txt'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    answer_path = os.path.join(script_dir, answer_file)
    guess_path = os.path.join(script_dir, guess_file)
    with open(answer_path, encoding="utf-8") as f:
        answers = [line.strip() for line in f if len(line.strip()) == 5]
    with open(guess_path, encoding="utf-8") as f:
        guesses = [line.strip() for line in f if len(line.strip()) == 5]
    return answers, list(set(guesses + answers))

# Load word frequencies from file, or assign uniform probability if not available
def load_word_frequencies(filename='word_freq.txt', answers=None):
    if answers is None:
        raise ValueError("The 'answers' argument must not be None.")
    try:
        with open(filename) as f:
            freqs = {}
            for line in f:
                word, freq = line.strip().split()
                freqs[word] = float(freq)
            total = sum(freqs.get(w, 1) for w in answers)
            return {w: freqs.get(w, 1) / total for w in answers}
    except (FileNotFoundError, ValueError):
        return {w: 1 / len(answers) for w in answers}


# Calculate expected entropy for a guess over the candidate set
def expected_entropy(guess, candidates, word_probs):
    feedback_groups = defaultdict(float)
    for word in candidates:
        fb = feedback_code(guess, word)
        feedback_groups[fb] += word_probs[word]
    entropy = 0
    for p in feedback_groups.values():
        if p > 0:
            entropy += p * math.log2(1 / p)
    return entropy

def diversity_score(word, used_letters):
    word_set = set(word)
    new_letters = word_set - used_letters
    repeated_count = len(word) - len(word_set)
    return len(new_letters) - repeated_count

@lru_cache(maxsize=10000)
def value_of_information_cached(guess, candidates_tuple):
    patterns = set()
    for word in candidates_tuple:
        patterns.add(feedback_code(guess, word))
    return len(patterns)

def value_of_information(guess, candidates):
    return value_of_information_cached(guess, tuple(candidates))

def get_feedback(guess, target):
    code = feedback_code(guess, target)
    # Decode the base-3 integer to string 'G', 'Y', 'B'
    mapping = {0: 'B', 1: 'Y', 2: 'G'}
    result = []
    for _ in range(5):
        result.append(mapping[code % 3])
        code //= 3
    return ''.join(result[::-1])

# Find the best guess in parallel by minimizing expected entropy
from concurrent.futures import as_completed

def parallel_best_guess(guess_list, candidates, word_probs, max_workers=None):
    best_guess = None
    best_entropy = float('inf')
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_guess = {executor.submit(expected_entropy, g, candidates, word_probs): g for g in guess_list}
        for future in as_completed(future_to_guess):
            guess = future_to_guess[future]
            entropy = future.result()
            if entropy < best_entropy:
                best_entropy = entropy
                best_guess = guess
    return best_guess

# Filter candidate words based on guess and feedback
def filter_candidates(candidates, guess, feedback):
    return [w for w in candidates if get_feedback(guess, w) == feedback]

# Pre-compute candidate sets for faster filtering
def filter_candidates_set(candidates_set, guess, feedback):
    return {w for w in candidates_set if get_feedback(guess, w) == feedback}

# GUI class for Wordle Solver
class WordleGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wordle Solver")
        self.answers, self.guesses = load_words()
        self.word_probs = load_word_frequencies(answers=self.answers)

        # Pre-compute numpy array of probabilities for faster entropy calculation
        self.word_probs_array = np.array([self.word_probs.get(word, 1/len(self.answers))
                                         for word in self.answers], dtype=np.float64)

        self.layout = QVBoxLayout()

        # Game grid
        self.grid = QGridLayout()
        self.labels = [[QLabel(" ") for _ in range(5)] for _ in range(6)]
        for r in range(6):
            for c in range(5):
                label = self.labels[r][c]
                label.setFixedSize(50, 50)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(
                    "border: 1px solid #888; font-size: 24px; background-color: #787c7e; color: white;"
                )
                self.grid.addWidget(label, r, c)
        self.layout.addLayout(self.grid)

        # Input row (guess + feedback)
        self.input_row = QGridLayout()
        self.input_label = QLabel("Guess:")
        self.input_row.addWidget(self.input_label, 0, 0)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter 5-letter guess (optional)")
        self.input_row.addWidget(self.input, 0, 1)
        self.feedback_label = QLabel("Feedback:")
        self.input_row.addWidget(self.feedback_label, 0, 2)
        self.feedback_input = QLineEdit()
        self.feedback_input.setPlaceholderText("e.g. GYBBY (optional)")
        self.input_row.addWidget(self.feedback_input, 0, 3)
        self.layout.addLayout(self.input_row)

        # Tactic selection buttons
        self.tactic_layout = QGridLayout()
        self.tactic_label = QLabel("Choose tactic:")
        self.tactic_layout.addWidget(self.tactic_label, 0, 0)

        self.random_button = QPushButton("Random")
        self.random_button.setCheckable(True)
        self.random_button.setChecked(True)
        self.random_button.clicked.connect(lambda: self.set_tactic("random"))
        self.tactic_layout.addWidget(self.random_button, 0, 1)

        self.entropy_button = QPushButton("Entropy")
        self.entropy_button.setCheckable(True)
        self.entropy_button.setChecked(False)
        self.entropy_button.clicked.connect(lambda: self.set_tactic("entropy"))
        self.tactic_layout.addWidget(self.entropy_button, 0, 2)
        self.diversity_button = QPushButton("Diversity")
        self.diversity_button.setCheckable(True)
        self.diversity_button.clicked.connect(lambda: self.set_tactic("diversity"))
        self.tactic_layout.addWidget(self.diversity_button, 0, 3)
        self.voi_button = QPushButton("Value of Information")
        self.voi_button.setCheckable(True)
        self.voi_button.clicked.connect(lambda: self.set_tactic("voi"))
        self.tactic_layout.addWidget(self.voi_button, 0, 4)
        self.layout.addLayout(self.tactic_layout)
        self.current_tactic = "random"

        # Solve, Restart, and Cancel buttons
        self.button_row = QGridLayout()
        self.button = QPushButton("Solve")
        self.button.clicked.connect(self.solve)
        self.button_row.addWidget(self.button, 0, 0)

        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart_game)
        self.button_row.addWidget(self.restart_button, 0, 1)

        self.cancel_button = QPushButton("Cancel calculation")
        self.cancel_button.clicked.connect(self.cancel_entropy)
        self.cancel_button.setVisible(False)
        self.button_row.addWidget(self.cancel_button, 0, 2)

        self.layout.addLayout(self.button_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        # Tables for suggestions
        self.suggestion_layout = QGridLayout()
        self.entropy_table = QTextEdit()
        self.entropy_table.setReadOnly(True)
        self.entropy_table.setMinimumWidth(250)
        self.entropy_table.setMaximumHeight(180)
        self.entropy_table.setPlaceholderText("Top 5 Entropy")
        self.suggestion_layout.addWidget(QLabel("Top 5 Entropy:"), 0, 0)
        self.suggestion_layout.addWidget(self.entropy_table, 1, 0)

        self.diversity_table = QTextEdit()
        self.diversity_table.setReadOnly(True)
        self.diversity_table.setMinimumWidth(250)
        self.diversity_table.setMaximumHeight(180)
        self.diversity_table.setPlaceholderText("Top 5 Diversity")
        self.suggestion_layout.addWidget(QLabel("Top 5 Diversity:"), 0, 1)
        self.suggestion_layout.addWidget(self.diversity_table, 1, 1)

        self.voi_table = QTextEdit()
        self.voi_table.setReadOnly(True)
        self.voi_table.setMinimumWidth(250)
        self.voi_table.setMaximumHeight(180)
        self.voi_table.setPlaceholderText("Top 5 VOI")
        self.suggestion_layout.addWidget(QLabel("Top 5 Value of Information:"), 0, 2)
        self.suggestion_layout.addWidget(self.voi_table, 1, 2)

        self.layout.addLayout(self.suggestion_layout)

        # Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.layout.addWidget(self.console)

        self.setLayout(self.layout)
        self.entropy_worker = None

        # State
        self.used_words = set()
        self.used_letters = set()
        self.candidates = self.answers[:]
        self.secret = ""
        self.turn = 0

        self.resize(900, 900)

    def set_tactic(self, tactic):
        self.current_tactic = tactic
        self.random_button.setChecked(tactic == "random")
        self.entropy_button.setChecked(tactic == "entropy")
        self.diversity_button.setChecked(tactic == "diversity")
        self.voi_button.setChecked(tactic == "voi")

    def restart_game(self):
        for r in range(6):
            for c in range(5):
                label = self.labels[r][c]
                label.setText(" ")
                label.setStyleSheet(
                    "border: 1px solid gray; font-size: 24px; background-color: #787c7e; color: white;"
                )
        self.input.clear()
        self.feedback_input.clear()
        self.console.clear()
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.entropy_table.clear()
        self.diversity_table.clear()
        self.voi_table.clear()
        self.used_words = set()
        self.used_letters = set()
        self.candidates = self.answers[:]
        self.secret = ""
        self.turn = 0

    def color_tile(self, label, feedback_char):
        if feedback_char == 'G':
            label.setStyleSheet("background-color: #6aaa64; font-size: 24px; color: white; font-weight: bold;")
        elif feedback_char == 'Y':
            label.setStyleSheet("background-color: #c9b458; font-size: 24px; color: white; font-weight: bold;")
        else:
            label.setStyleSheet("background-color: #787c7e; font-size: 24px; color: white; font-weight: bold;")

    def cancel_entropy(self):
        if self.entropy_worker is not None:
            self.entropy_worker.cancel()

    def solve(self):
        # If first turn, set random secret
        if self.turn == 0:
            self.secret = random.choice(self.answers)
            self.console.append(f"Solving for: {self.secret.upper()}")
            self.candidates = self.answers[:]
            self.used_words = set()
            self.used_letters = set()
            self.entropy_table.clear()
            self.diversity_table.clear()
            self.voi_table.clear()
            for r in range(6):
                for c in range(5):
                    self.labels[r][c].setText(" ")
                    self.labels[r][c].setStyleSheet(
                        "border: 1px solid gray; font-size: 24px; background-color: #787c7e; color: white;"
                    )

        guess_input = self.input.text().strip().lower()
        feedback_input = self.feedback_input.text().strip().upper()
        guess = None
        feedback = None

        # If user provided both guess and feedback, use them
        if guess_input and feedback_input and len(guess_input) == 5 and len(feedback_input) == 5:
            guess = guess_input
            if guess not in set(self.guesses).union(self.answers) or guess in self.used_words:
                self.console.append("Invalid or repeated guess.")
                return

            # Show guess and feedback in grid
            self.used_words.add(guess)
            self.used_letters.update(guess)
            self.console.append(f"Turn {self.turn+1}: {guess.upper()} -> {feedback_input}")
            for i in range(5):
                self.labels[self.turn][i].setText(guess[i].upper())
                self.color_tile(self.labels[self.turn][i], feedback_input[i])

            if feedback_input == "GGGGG":
                self.console.append("Solved!")
                self.console.append("Click 'Restart' to play again.")
                self.turn += 1
                return

            self.candidates = filter_candidates(self.candidates, guess, feedback_input)
            self.turn += 1

            if self.turn >= 6:
                self.console.append("Failed to solve.")

            self.input.clear()
            self.feedback_input.clear()
            return

        # Random tactic
        if self.current_tactic == "random":
            available = [w for w in self.candidates if w not in self.used_words]
            if not available:
                self.console.append("No more guesses available.")
                return
            guess = random.choice(available)
            feedback = get_feedback(guess, self.secret)
            self.used_words.add(guess)
            self.used_letters.update(guess)
            self.console.append(f"Turn {self.turn+1}: {guess.upper()} -> {feedback}")

            for i in range(5):
                self.labels[self.turn][i].setText(guess[i].upper())
                self.color_tile(self.labels[self.turn][i], feedback[i])

            if feedback == "GGGGG":
                self.console.append("Solved!")
                self.console.append("Click 'Restart' to play again.")
                self.turn += 1
                return

            self.candidates = filter_candidates(self.candidates, guess, feedback)
            self.turn += 1

            if self.turn >= 6:
                self.console.append("Failed to solve.")

            self.input.clear()
            self.feedback_input.clear()
            # Clear suggestion tables for random mode
            self.entropy_table.clear()
            self.diversity_table.clear()
            self.voi_table.clear()
            return

        # Calculate entropy for guess pool
        if self.turn < 3:
            guess_pool = [w for w in self.guesses if w not in self.used_words]
        else:
            guess_pool = [w for w in self.candidates if w not in self.used_words]

        if not guess_pool:
            self.console.append("No more guesses available.")
            return

        # Progress bar and cancellation
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        self.console.append(f"Calculating entropy for {len(guess_pool)} words...")

        entropy_threshold = None
        entropies_result = {}
        finished_flag = [False]

        def on_progress(val):
            self.progress_bar.setValue(val)

        def on_finished(entropies):
            entropies_result.update(entropies)
            finished_flag[0] = True

        self.entropy_worker = EntropyWorker(guess_pool, self.candidates, self.word_probs, entropy_threshold)
        self.entropy_worker.progress.connect(on_progress)
        self.entropy_worker.finished.connect(on_finished)

        def on_cancelled():
            finished_flag[0] = True
            self.console.append("Calculation cancelled.")
            self.progress_bar.setVisible(False)
            self.cancel_button.setVisible(False)

        self.entropy_worker.cancelled.connect(on_cancelled)

        event_loop = QEventLoop()
        def finish_event_loop(*args, **kwargs):
            if event_loop.isRunning():
                event_loop.quit()
        self.entropy_worker.finished.connect(finish_event_loop)
        self.entropy_worker.cancelled.connect(finish_event_loop)

        self.entropy_worker.start()
        event_loop.exec()

        entropies = entropies_result
        voi_scores = {word: value_of_information(word, self.candidates) for word in guess_pool}
        diversity_scores = {word: diversity_score(word, self.used_letters) for word in guess_pool}

        # Selection logic based on tactic
        if self.current_tactic == "entropy":
            entropy_top5 = sorted(entropies.items(), key=lambda x: x[1])[:5]
            if not entropy_top5:
                self.console.append("No entropy results available.")
                return
            min_entropy = entropy_top5[0][1]
            best_words = [w for w, e in entropies.items() if abs(e - min_entropy) < 1e-8]
            if len(best_words) > 1:
                guess = max(best_words, key=lambda w: diversity_scores[w])
            else:
                guess = best_words[0]
        elif self.current_tactic == "diversity":
            diversity_top5 = sorted(guess_pool, key=lambda w: (-diversity_scores[w], entropies[w]))[:5]
            if not diversity_top5:
                self.console.append("No diversity candidates available.")
                return
            max_div = diversity_scores[diversity_top5[0]]
            best_diverse = [w for w, d in diversity_scores.items() if d == max_div]
            min_entropy = min(entropies[w] for w in best_diverse)
            best_words = [w for w in best_diverse if abs(entropies[w] - min_entropy) < 1e-8]
            if len(best_words) > 1:
                guess = max(best_words, key=lambda w: voi_scores[w])
            else:
                guess = best_words[0]
        elif self.current_tactic == "voi":
            voi_top5 = sorted(guess_pool, key=lambda w: (-voi_scores[w], entropies[w]))[:5]
            if not voi_top5:
                self.console.append("No VOI results available.")
                return
            max_voi = voi_scores[voi_top5[0]]
            best_voi = [w for w, v in voi_scores.items() if v == max_voi]
            min_entropy = min(entropies[w] for w in best_voi)
            best_words = [w for w in best_voi if abs(entropies[w] - min_entropy) < 1e-8]
            if len(best_words) > 1:
                guess = max(best_words, key=lambda w: diversity_scores[w])
            else:
                guess = best_words[0]

        # Fill suggestion tables
        # Entropy
        entropy_top5 = sorted(entropies.items(), key=lambda x: x[1])[:5]
        self.entropy_table.clear()
        for i, (word, entropy_val) in enumerate(entropy_top5):
            prefix = "-> " if self.current_tactic == "entropy" and word == guess and i == 0 else ""
            self.entropy_table.append(f"{prefix}{i+1}. {word.upper()} (Entropy: {entropy_val:.2f})")
        # Diversity
        diversity_top5 = sorted(guess_pool, key=lambda w: (-diversity_scores[w], entropies[w]))[:5]
        self.diversity_table.clear()
        for i, word in enumerate(diversity_top5):
            prefix = "-> " if self.current_tactic == "diversity" and word == guess and i == 0 else ""
            self.diversity_table.append(f"{prefix}{i+1}. {word.upper()} (Diversity: {diversity_scores[word]}, Entropy: {entropies[word]:.2f})")
        # VOI
        voi_top5 = sorted(guess_pool, key=lambda w: (-voi_scores[w], entropies[w]))[:5]
        self.voi_table.clear()
        for i, word in enumerate(voi_top5):
            prefix = "-> " if self.current_tactic == "voi" and word == guess and i == 0 else ""
            self.voi_table.append(f"{prefix}{i+1}. {word.upper()} (VOI: {voi_scores[word]}, Entropy: {entropies[word]:.2f})")

        # Show guess and feedback in grid
        feedback = get_feedback(guess, self.secret)
        self.used_words.add(guess)
        self.used_letters.update(guess)
        self.console.append(f"Turn {self.turn+1}: {guess.upper()} -> {feedback}")

        for i in range(5):
            self.labels[self.turn][i].setText(guess[i].upper())
            self.color_tile(self.labels[self.turn][i], feedback[i])

        if feedback == "GGGGG":
            self.console.append("Solved!")
            self.console.append("Click 'Restart' to play again.")
            self.turn += 1
            return

        self.candidates = filter_candidates(self.candidates, guess, feedback)
        self.turn += 1

        if self.turn >= 6:
            self.console.append("Failed to solve.")

        # Clear guess/feedback input for next turn

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    gui = WordleGUI()
    gui.resize(420, 700)
    gui.show()
    sys.exit(app.exec())
