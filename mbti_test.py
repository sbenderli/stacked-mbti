import json
import random
import itertools
import uuid
import logging
import math
from flask import Flask, render_template_string, request, redirect, url_for, jsonify

# Configure logging at the DEBUG level
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = "CHANGE_THIS_TO_A_SECRET"

# ===== CONFIGURABLE CONSTANTS =====
NUM_QUESTIONS = 16                # Total number of questions in the test
PROMPTS_PER_QUESTION = 4          # Number of prompts per question
FUNCTIONS = ["Te", "Ti", "Fe", "Fi", "Ne", "Ni", "Se", "Si"]

# TARGET_COUNT: Each function appears exactly this many times.
TARGET_COUNT = (NUM_QUESTIONS * PROMPTS_PER_QUESTION) // len(FUNCTIONS)

POINTS_CONFIG = [3, 2, 1, 0]  # Points for positions 1st to 4th (used during scoring)

PROMPTS_FILE = "prompts.json"

# ===== MBTI TYPE TO FUNCTIONAL STACK MAPPING =====
MBTI_STACKS = {
    "ENFP": ["Ne", "Fi", "Te", "Si"],
    "ENFJ": ["Fe", "Ni", "Se", "Ti"],
    "ENTJ": ["Te", "Ni", "Se", "Fi"],
    "ENTP": ["Ne", "Ti", "Fe", "Si"],
    "INFP": ["Fi", "Ne", "Si", "Te"],
    "INFJ": ["Ni", "Fe", "Ti", "Se"],
    "INTJ": ["Ni", "Te", "Fi", "Se"],
    "INTP": ["Ti", "Ne", "Si", "Fe"],
    "ESFP": ["Se", "Fi", "Te", "Ni"],
    "ESTP": ["Se", "Ti", "Fe", "Ni"],
    "ESFJ": ["Fe", "Si", "Ne", "Ti"],
    "ESTJ": ["Te", "Si", "Ne", "Fi"],
    "ISFP": ["Fi", "Se", "Ni", "Te"],
    "ISTP": ["Ti", "Se", "Ni", "Fe"],
    "ISFJ": ["Si", "Fe", "Ti", "Ne"],
    "ISTJ": ["Si", "Te", "Fi", "Ne"]
}

# ===== LOAD PROMPTS FROM JSON FILE =====
with open(PROMPTS_FILE, "r") as f:
    PROMPTS_DB = json.load(f)

# ===== GENERATE A BALANCED ASSIGNMENT MATRIX =====
def generate_assignment_matrix():
    """
    Generates a list of NUM_QUESTIONS groups (one per question) where each group is a list 
    of PROMPTS_PER_QUESTION function names. Each of the FUNCTIONS appears exactly TARGET_COUNT times.
    """
    assignment = [None] * NUM_QUESTIONS  # Each assignment[q] is a list of functions for question q.
    counts = {f: 0 for f in FUNCTIONS}

    def backtrack(q_index):
        if q_index == NUM_QUESTIONS:
            return all(counts[f] == TARGET_COUNT for f in FUNCTIONS)
        # Choose any combination of PROMPTS_PER_QUESTION functions (order doesn't matter)
        combs = list(itertools.combinations(FUNCTIONS, PROMPTS_PER_QUESTION))
        random.shuffle(combs)
        for comb in combs:
            if any(counts[f] >= TARGET_COUNT for f in comb):
                continue
            for f in comb:
                counts[f] += 1
            assignment[q_index] = list(comb)
            # Prune: ensure that remaining questions can still fulfill TARGET_COUNT for each function.
            remaining_questions = NUM_QUESTIONS - q_index - 1
            if all((TARGET_COUNT - counts[f]) <= remaining_questions for f in FUNCTIONS):
                if backtrack(q_index + 1):
                    return True
            for f in comb:
                counts[f] -= 1
        return False

    if backtrack(0):
        return assignment
    else:
        raise Exception("Could not generate a balanced assignment matrix.")

# ===== GENERATE THE TEST =====
def generate_test(seed=None):
    """
    Creates a test JSON object containing:
      - test_id: a UUID for the test.
      - seed: the seed used for randomization.
      - questions: a list of NUM_QUESTIONS questions. Each question is a dict with:
          - question_number (1-indexed)
          - prompts: a list of PROMPTS_PER_QUESTION prompt dicts (each with id, function, and text) 
                     chosen as 2 diametrically opposing pairs. The 4 prompts are randomly shuffled before display.
          - user_ranking: (initially empty) to store the ranking order.
      - function_scores: will be computed at the end.
      
    The selection for each question works as follows:
      a) There are four diametric pairs: (Te, Fi), (Ti, Fe), (Ne, Si), and (Ni, Se).
      b) Each pair appears exactly TARGET_PAIRS times overall, where TARGET_PAIRS = NUM_QUESTIONS/2.
      c) For each question, two pairs are randomly chosen (without replacement).
         For each pair, a coin flip decides which function is used first.
      d) For each chosen function, a prompt is popped from that function's pool.
      e) The resulting 4 prompts are then randomly shuffled before being displayed.
    """
    if seed is None:
        seed = random.randrange(1, 10**9)
    else:
        try:
            seed = int(seed)
        except ValueError:
            seed = random.randrange(1, 10**9)
    
    logging.debug("Using seed: %s", seed)
    random.seed(seed)
    test_id = str(uuid.uuid4())

    # Prepare the prompt pool for each function.
    prompt_assignment = {}
    for func in FUNCTIONS:
        prompt_list = []
        for prompt_id, prompt_text in PROMPTS_DB[func].items():
            prompt_list.append({"id": prompt_id, "function": func, "text": prompt_text})
        random.shuffle(prompt_list)
        prompt_assignment[func] = prompt_list

    # Define the four diametric pairs.
    diametric_pairs = [
        ("Te", "Fi"),
        ("Ti", "Fe"),
        ("Ne", "Si"),
        ("Ni", "Se")
    ]
    # Each pair must appear exactly TARGET_PAIRS times.
    TARGET_PAIRS = (NUM_QUESTIONS * 2) // len(diametric_pairs)  # = NUM_QUESTIONS/2
    pair_list = []
    for pair in diametric_pairs:
        for _ in range(TARGET_PAIRS):
            pair_list.append(pair)
    random.shuffle(pair_list)

    # Create questions by assigning 2 pairs (4 prompts) per question.
    questions = []
    for qi in range(NUM_QUESTIONS):
        # Pop two pairs from the shuffled pair_list.
        pair1 = pair_list.pop()
        pair2 = pair_list.pop()
        # For each pair, randomly decide the order.
        if random.choice([True, False]):
            first1, second1 = pair1
        else:
            first1, second1 = pair1[1], pair1[0]
        if random.choice([True, False]):
            first2, second2 = pair2
        else:
            first2, second2 = pair2[1], pair2[0]
        # For each function, pop a prompt from its pool.
        prompt1 = prompt_assignment[first1].pop(0)
        prompt2 = prompt_assignment[second1].pop(0)
        prompt3 = prompt_assignment[first2].pop(0)
        prompt4 = prompt_assignment[second2].pop(0)
        # Assemble the list of 4 prompts and randomly shuffle them.
        prompts = [prompt1, prompt2, prompt3, prompt4]
        random.shuffle(prompts)
        # Assemble the question.
        question = {
            "question_number": qi + 1,
            "prompts": prompts,
            "user_ranking": []  # to be filled in when the user ranks the prompts
        }
        questions.append(question)

    test = {
        "test_id": test_id,
        "seed": seed,
        "questions": questions,
        "function_scores": {}
    }
    return test

# ===== GLOBAL STORAGE FOR TESTS (for demonstration) =====
TESTS = {}

# ===== PREDICTION METHODS (unchanged) =====

def ideal_weight(f, mbti):
    stack = MBTI_STACKS[mbti]
    if f in stack:
        pos = stack.index(f)
        return 3 - pos  # positions: 0 -> 3, 1 -> 2, 2 -> 1, 3 -> 0
    else:
        return -1

def predict_types_weighted_distance(function_scores):
    predictions = {}
    for mbti, stack in MBTI_STACKS.items():
        error = 0
        for f in FUNCTIONS:
            ideal = ideal_weight(f, mbti)
            error += (function_scores.get(f, 0) - ideal) ** 2
        predictions[mbti] = error
    sorted_types = sorted(predictions.items(), key=lambda x: x[1])
    return sorted_types[:3]

def predict_types_rank_correlation(function_scores):
    sorted_funcs = sorted(FUNCTIONS, key=lambda f: -function_scores.get(f, 0))
    observed_ranks = {f: i+1 for i, f in enumerate(sorted_funcs)}
    predictions = {}
    for mbti, stack in MBTI_STACKS.items():
        diff = 0
        for ideal_rank, f in enumerate(stack, start=1):
            diff += abs(observed_ranks[f] - ideal_rank)
        predictions[mbti] = diff
    sorted_types = sorted(predictions.items(), key=lambda x: x[1])
    return sorted_types[:3]

def predict_types_pairwise(function_scores):
    predictions = {}
    for mbti, stack in MBTI_STACKS.items():
        count = 0
        for i in range(len(stack)):
            for j in range(i+1, len(stack)):
                f1 = stack[i]
                f2 = stack[j]
                if function_scores.get(f1, 0) > function_scores.get(f2, 0):
                    count += 1
        predictions[mbti] = count  # Maximum is 6
    sorted_types = sorted(predictions.items(), key=lambda x: -x[1])
    return sorted_types[:3]

def predict_types_bayesian(function_scores, sigma=5):
    predictions = {}
    for mbti, stack in MBTI_STACKS.items():
        log_likelihood = 0
        for f in FUNCTIONS:
            mu = ideal_weight(f, mbti)
            obs = function_scores.get(f, 0)
            log_likelihood += -((obs - mu) ** 2) / (2 * sigma * sigma)
        predictions[mbti] = log_likelihood
    sorted_types = sorted(predictions.items(), key=lambda x: -x[1])
    return sorted_types[:3]

def compare_predictions(function_scores):
    return {
        "weighted_distance": predict_types_weighted_distance(function_scores),
        "rank_correlation": predict_types_rank_correlation(function_scores),
        "pairwise": predict_types_pairwise(function_scores),
        "bayesian": predict_types_bayesian(function_scores)
    }

def compute_meta_prediction(predictions):
    """
    Combines predictions from each method (weighted_distance, rank_correlation, pairwise, bayesian)
    by awarding points (3 for #1, 2 for #2, 1 for #3) for each method's top 3.
    
    Instead of considering all types, this version only keeps the top 2 types (with the highest
    total points) and then calculates a probability for each, such that their sum is 100%.
    
    Returns:
      - meta_prediction: the type with the highest meta score (among the top 2)
      - meta_probabilities: a dict with the two types as keys and their probability (as a fraction) as values.
      - meta_scores: a dict of the meta scores for the top 2 types.
    """
    meta_scores = {}
    # Sum points for each type over all methods:
    for method, top3 in predictions.items():
        for rank, (mbti, _) in enumerate(top3, start=1):
            points = 4 - rank  # Rank 1 gets 3 points, 2 gets 2, 3 gets 1.
            meta_scores[mbti] = meta_scores.get(mbti, 0) + points

    # Sort the meta scores in descending order and take the top 2.
    sorted_meta = sorted(meta_scores.items(), key=lambda x: -x[1])
    top2 = sorted_meta[:2]
    
    if len(top2) < 2:
        # In the unlikely event there's only one type, return it with probability 1.0.
        top2 = sorted_meta
    
    total_points = top2[0][1] + top2[1][1]
    meta_probabilities = {mbti: score / total_points for mbti, score in top2}
    meta_prediction = top2[0][0]
    
    return meta_prediction, meta_probabilities, dict(top2)

def compute_results(test):
    scores = {f: 0 for f in FUNCTIONS}
    for q in test["questions"]:
        ranking = q.get("user_ranking", [])
        for pos, prompt_id in enumerate(ranking):
            try:
                points = POINTS_CONFIG[pos]
            except IndexError:
                points = 0
            func = prompt_id.split("-")[0]
            scores[func] += points
    test["function_scores"] = scores
    return test

# ===== FLASK ROUTES & TEMPLATES =====

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>MBTI Stacking Test Instructions</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
    button { font-size: 18px; padding: 10px 20px; }
  </style>
</head>
<body>
  <h1>Welcome to the MBTI Stacking Test</h1>
  <p>Your test consists of {{ NUM_QUESTIONS }} questions. Each question presents {{ PROMPTS_PER_QUESTION }} prompts (each from a different cognitive function).
  Once a prompt is used, it will not appear again.</p>
  <p>If you want a reproducible test, pass a seed value in the URL (e.g., <code>/start?seed=12345</code>).</p>
  <button onclick="window.location.href='/start'">Start Test</button>
  <br><br>
  <button onclick="window.location.href='/predict'">View Sample Predictions</button>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, NUM_QUESTIONS=NUM_QUESTIONS, PROMPTS_PER_QUESTION=PROMPTS_PER_QUESTION)

@app.route("/start")
def start():
    seed = request.args.get("seed", None)
    test = generate_test(seed)
    TESTS[test["test_id"]] = test
    return redirect(url_for("question", qid=1, test_id=test["test_id"]))

QUESTION_HTML = """
<!doctype html>
<html>
<head>
  <title>Question {{ question.question_number }} of {{ NUM_QUESTIONS }}</title>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://code.jquery.com/ui/1.13.2/jquery-ui.min.js"></script>
  <link rel="stylesheet" href="https://code.jquery.com/ui/1.13.2/themes/base/jquery-ui.css">
  <style>
    body { font-family: Arial, sans-serif; text-align: center; margin-top: 30px; }
    .prompt-list { list-style-type: none; padding: 0; width: 50%; margin: 0 auto; }
    .prompt-item { margin: 10px; padding: 20px; border: 1px solid #ccc; background: #f9f9f9; cursor: move; font-size: 18px; }
    button { font-size: 16px; padding: 10px 15px; margin: 20px; }
  </style>
  <script>
    $(function() {
      $("#sortable").sortable();
      $("#sortable").disableSelection();
      $("#answerForm").submit(function() {
        var order = [];
        $("#sortable li").each(function(){
          order.push($(this).attr("data-prompt-id"));
        });
        $("#ranking").val(order.join(","));
        return true;
      });
    });
  </script>
</head>
<body>
  <h2>Question {{ question.question_number }} of {{ NUM_QUESTIONS }}</h2>
  <p>Drag and drop the boxes below to rank them from <strong>most preferred</strong> to <strong>least preferred</strong>.</p>
  <form id="answerForm" method="POST" action="{{ url_for('submit_answer') }}">
    <ul id="sortable" class="prompt-list">
      {% for prompt in question.prompts %}
      <li class="prompt-item" data-prompt-id="{{ prompt.id }}">
        {% if debug %}
          <strong>{{ prompt.function }}</strong>: 
        {% endif %}
        {{ prompt.text }}
      </li>
      {% endfor %}
    </ul>
    <input type="hidden" name="ranking" id="ranking" value="">
    <input type="hidden" name="qid" value="{{ question.question_number }}">
    <input type="hidden" name="test_id" value="{{ test_id }}">
    <div>
      {% if question.question_number > 1 %}
      <button type="button" onclick="window.location.href='{{ url_for('question', qid=question.question_number-1, test_id=test_id) }}'">Previous</button>
      {% endif %}
      <button type="submit">
        {% if question.question_number == NUM_QUESTIONS %}Finish{% else %}Next{% endif %}
      </button>
    </div>
  </form>
</body>
</html>
"""

@app.route("/question/<int:qid>")
def question(qid):
    test_id = request.args.get("test_id")
    if not test_id or test_id not in TESTS:
        return "Invalid test session.", 400
    test = TESTS[test_id]
    if not (1 <= qid <= NUM_QUESTIONS):
        return "Question number out of range.", 400
    question = test["questions"][qid - 1]
    return render_template_string(QUESTION_HTML, question=question, test_id=test_id, debug=app.debug, NUM_QUESTIONS=NUM_QUESTIONS)

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    test_id = request.form.get("test_id")
    if not test_id or test_id not in TESTS:
        return "Invalid test session.", 400
    test = TESTS[test_id]
    try:
        qid = int(request.form.get("qid"))
    except (TypeError, ValueError):
        return "Invalid question number.", 400
    ranking_str = request.form.get("ranking", "")
    ranking = ranking_str.split(",") if ranking_str else []
    if not (1 <= qid <= NUM_QUESTIONS):
        return "Question number out of range.", 400
    test["questions"][qid - 1]["user_ranking"] = ranking
    if qid < NUM_QUESTIONS:
        return redirect(url_for("question", qid=qid + 1, test_id=test_id))
    else:
        compute_results(test)
        # Save the final test JSON to a file named {test_id}.json
        with open(f"results/{test_id}.json", "w") as outfile:
            json.dump(test, outfile, indent=2)
        return redirect(url_for("final_results", test_id=test_id))

@app.route("/results")
def results():
    test_id = request.args.get("test_id")
    if not test_id or test_id not in TESTS:
        return "Invalid test session.", 400
    test = TESTS[test_id]
    return jsonify(test)

@app.route("/final_results")
def final_results():
    test_id = request.args.get("test_id")
    if not test_id or test_id not in TESTS:
        return "Invalid test session.", 400
    test = TESTS[test_id]
    function_scores = test.get("function_scores", {})
    
    # Compute the predictions using the various methods.
    predictions = compare_predictions(function_scores)
    meta_prediction, meta_probabilities, meta_scores = compute_meta_prediction(predictions)
    
    # Build a prediction dictionary.
    prediction = {
        "meta_prediction": meta_prediction,
        "meta_probabilities": meta_probabilities,
        "meta_scores": meta_scores,
        "predictions": predictions
    }
    
    # Store the prediction into the test JSON under a new field.
    test["prediction"] = prediction

    explanation = {
        "weighted_distance": "This method compares your function scores to an ideal profile for each MBTI type. It then ranks the types by how close your scores are to the ideal.",
        "rank_correlation": "This method ranks your functions from highest to lowest and compares that order to the expected order for each type. The closer the match, the higher the ranking.",
        "pairwise": "This method looks at pairs of functions (for example, is your dominant function higher than your auxiliary?) and counts how many of these comparisons match each type's ideal ordering.",
        "bayesian": "This method uses a probability model to estimate how likely it is that your scores fit the ideal profile of each MBTI type."
    }
    
    FINAL_RESULTS_HTML = """
    <!doctype html>
    <html>
    <head>
      <title>Your MBTI Stacking Test Results</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        h1, h2, h3 { color: #333; }
        .section { margin-bottom: 30px; }
        .method { border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
      </style>
    </head>
    <body>
      <h1>Your MBTI Stacking Test Results</h1>
      
      <div class="section">
        <h2>What We Measured</h2>
        <p>This test measured your functional preferences across eight cognitive functions: Te, Ti, Fe, Fi, Ne, Ni, Se, and Si. Below are your raw function scores:</p>
        <ul>
          {% for func, score in function_scores.items() %}
          <li><strong>{{ func }}</strong>: {{ score }}</li>
          {% endfor %}
        </ul>
      </div>
      
      <div class="section">
        <h2>Method Predictions</h2>
        <p>We used four different methods to compare your scores to the ideal profiles for each MBTI type. Here is a brief explanation of each method and the top 3 types it predicted (in order):</p>
        
        {% for method, prediction in predictions.items() %}
        <div class="method">
          <h3>{{ method.replace('_', ' ').title() }}</h3>
          <p>{{ explanation[method] }}</p>
          <ol>
            {% for mbti, _ in prediction %}
            <li>{{ mbti }}</li>
            {% endfor %}
          </ol>
        </div>
        {% endfor %}
      </div>
      
      <div class="section">
        <h2>Combined (Meta) Prediction</h2>
        <p>We combined the predictions from each method by awarding points to each type (3 points for a #1 ranking, 2 points for #2, and 1 point for #3). We then restricted our results to only the top 2 candidates and computed the probabilities accordingly. Your most likely type is:</p>
        <h3>{{ meta_prediction }}</h3>
        <p>Meta Prediction Breakdown (only the top two types are considered):</p>
        <ul>
          {% for mbti, prob in meta_probabilities.items() %}
          <li>{{ mbti }}: {{ (prob * 100) | round(1) }}% ({{ meta_scores[mbti] }} points)</li>
          {% endfor %}
        </ul>
      </div>
      
      <div class="section">
        <p><em>Note:</em> The results above are based on several different approaches to comparing your function scores to ideal profiles. Over time, we plan to enhance these with additional charts and interactive features.</p>
      </div>
    </body>
    </html>
    """
    return render_template_string(FINAL_RESULTS_HTML,
                                  function_scores=function_scores,
                                  predictions=predictions,
                                  explanation=explanation,
                                  meta_prediction=meta_prediction,
                                  meta_probabilities=meta_probabilities,
                                  meta_scores=meta_scores)

@app.route("/predict")
def predict():
    sample_scores = {
        "Fe": 20,
        "Fi": 16,
        "Ne": 17,
        "Ni": 12,
        "Se": 7,
        "Si": 3,
        "Te": 12,
        "Ti": 9
    }
    predictions = compare_predictions(sample_scores)
    meta_prediction, meta_probabilities, meta_scores = compute_meta_prediction(predictions)
    return jsonify({
        "sample_function_scores": sample_scores,
        "predictions": predictions,
        "meta_prediction": meta_prediction,
        "meta_probabilities": meta_probabilities,
        "meta_scores": meta_scores
    })

if __name__ == "__main__":
    app.run(debug=False)