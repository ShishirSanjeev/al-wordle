## Introduction
In [Keith A. Brown's paper](https://www.nature.com/articles/s41524-022-00787-7), active learning strategies are described by means of the game, wordle. Taking inspiration from the paper, a GUI was built for the game wordle, using PySide6, which houses a few active learning strategies alongside for anyone to test out. 


## Active Learning strategies
### 1. Random
Selects a random word from the remaining candidate pool.

### 2. Entropy (Information Gain)
Minimizes expected entropy (information gain) by choosing words that provide the most information about remaining candidates. It calculates the expected reduction in uncertainty for each possible guess across all candidate words using Shannon entropy formula: `H = Σ(p_i * log₂(1/p_i))`

Note: Entropy calculations scale with guess pool size (typically 1000-10000 words)
### 3. Diversity
Maximizes letter diversity by prioritizing words with new letters not yet used. It scores words based on unique letters minus repeated letters, favoring exploration of new letter combinations.
### 4. Value of Information (VOI)
Maximizes the number of unique feedback patterns a guess can produce. It counts distinct feedback combinations (Green/Yellow/Black patterns) for each guess against candidates.



## Installation 
### Prerequisites:

- Python 3.8+
- PySide6
- NumPy
### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wordle-solver-al.git
cd wordle-solver-al
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Ensure data files are present:
   - `answers.txt`: List of valid Wordle answers
   - `guesses.txt`: List of valid guess words
   - `word_freq.txt` (optional): Word frequency data for probability weighting

  

## Usage

1. Run the application:
```bash
python wordle_solver.py
```


![[gui.png]]

2. The app will automatically select a random secret word to solve
3. Choose a strategy using the tactic buttons:
   - **Random**: Instant selection
   - **Entropy**: May take several seconds with progress bar
   - **Diversity**: Fast calculation
   - **VOI**: Fast calculation

2. Click "Solve" to let the AI make a guess, or enter your own guess and feedback

3. The grid will update with colors (Green/Yellow/Gray) matching Wordle

4. Continue until solved or 6 attempts reached

### Manual Input

- Enter a 5-letter guess in the "Guess" field

- Enter feedback as G/Y/B string (e.g., "GYBBY") in the "Feedback" field

- Click "Solve" to apply

  
