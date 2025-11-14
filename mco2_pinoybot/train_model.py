"""
train_model.py
PinoyBot - Model Training Script
--------------------------------
Trains a Decision Tree classifier for word-level language identification
(ENG, FIL, OTH) on code-switched Filipino text.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import optuna
from optuna.samplers import GridSampler

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(word):
    """Extract handcrafted features from a given word."""
    if not isinstance(word, str):
        word = str(word) if word is not None else ""
    
    w = word.lower()
    
    # Pre-compile regex patterns for better performance
    consonant_pattern = r'[bcdfghjklmnpqrstvwxyz]'
    consonant_clusters = len(re.findall(consonant_pattern + r'{2,}', w))
    starts_consonant_cluster = int(bool(re.match(consonant_pattern + r'{2,}', w)))
    
    # Calculate vowel and consonant counts
    vowel_count = sum(c in 'aeiou' for c in w)
    consonant_count = sum(c in 'bcdfghjklmnpqrstvwxyz' for c in w)
    
    return {
        "length": len(w),
        "vowel_ratio": vowel_count / max(1, len(w)),
        "consonant_clusters": consonant_clusters,
        "starts_consonant_cluster": starts_consonant_cluster,
        
        # FIL features
        "has_-": int(bool(re.search(r'[mnp]ag-', w))),
        "has_ng": int("ng" in w), # common in Filipino: "ng", "mang", "sang"
        "has_mga": int("mga" in w), # common in Filipino: "mga"
        "prefix_mag": int(w.startswith("mag")), # common in Filipino: "mag"
        "prefix_nag": int(w.startswith("nag")), # common in Filipino: "nag"
        "prefix_pag": int(w.startswith("pag")), # common in Filipino: "pag"
        "prefix_ka": int(w.startswith("ka") and len(w) > 2), # common in Filipino: "ka"
        "suffix_an": int(w.endswith("an") and len(w) > 2), # common in Filipino: "an"
        "suffix_in": int(w.endswith("in") and len(w) > 2), # common in Filipino: "in"
        "suffix_han": int(w.endswith("han")), # common in Filipino: "han"
        "suffix_o": int(w.endswith("o") and len(w) > 2), # common in Filipino: gusto, pero, ako, ko, ito, tao
        "has_uso": int("uso" in w), # common in Filipino: puso, kapuso, dugo, grupo
        "has_sto": int("sto" in w), # common in Filipino borrowed words: gusto, wasto, Agosto, gastos
        "starts_gu": int(w.startswith("gu")), # common Filipino verbs: gusto, gumawa, gumagawa, gumamit, gumising
        "has_abe": int("abe" in w),
        "pattern_cuco": int(bool(re.search(r'[bdghjklmnpqrstvwxyz]u[bdghjklmnpqrstvwxyz]o$', w))), # gusto, puso, turo, luto
        
        # ENG features
        "pattern_ix": int(bool(re.search(r'[i][snft]', w)) and len(w) < 3), # is in if it
        "ends_with_o_<2": int(bool(re.search(r'[stgd]o', w)) and len(w) < 3), # so go to do
        "suffix_ing": int(w.endswith("ing")), # learning, programming, procastinating LOOOOLLL
        "suffix_ed": int(w.endswith("ed")), #cooked, depressed, mega-cooked...
        "suffix_ly": int(w.endswith("ly")), # lovely, quickly, happily
        "suffix_tion": int(w.endswith("tion")), # nation, station, relation
        "suffix_ness": int(w.endswith("ness")), # happiness, sadness, kindness
        "suffix_er": int(w.endswith("er")), # teacher, player, runner, 
        "suffix_est": int(w.endswith("est")), # biggest, fastest, strongest
        "suffix_s": int(w.endswith("s") and len(w) > 2),  # English plurals
        "suffix_y": int(w.endswith("y") and len(w) > 2),  # today, yesterday, happy
        "prefix_un": int(w.startswith("un")), # unhappy, undone
        "prefix_re": int(w.startswith("re")), # redo, rewrite, reread
        "prefix_pre": int(w.startswith("pre")), # preview, pretest, pre-enlist
        "prefix_dis": int(w.startswith("dis")), # disconnect, disapprove
        "has_ed": int(w.endswith("ed")),   # cooked, played, jumped
        "has_ion": int("ion" in w), # nation, station, relation
        "has_tion": int("tion" in w), # nation, station, relation
        "has_th": int("th" in w), # common in English: "the", "this", "that"
        "has_sh": int("sh" in w), # common in English: "she", "shut", "ship"
        "has_ch": int("ch" in w), # common in English: "chocolate", "church", "check"
        "has_ph": int("ph" in w), # common in English: "phone", "photo", "graph"
        "has_qu": int("qu" in w), # common in English: "quick", "question", "quilt"
        "has_ck": int("ck" in w),  # back, check, quick
        "has_gh": int("gh" in w),  # night, through
        "has_wh": int("wh" in w),  # what, where, when
        "has_oo": int("oo" in w),  # good, food, book
        "has_ll": int("ll" in w),  # hello
        "has_ea": int("ea" in w),  # weather, beautiful, eat
        "has_ee": int("ee" in w),  # see, tree, meet, feel 
        "has_ou": int("ou" in w),  # house, mouse, out, about
        "has_ai": int("ai" in w),  # rain, wait, main
        "starts_str": int(w.startswith("str")),  # street, string, strong 
        "starts_spr": int(w.startswith("spr")),  # spring, spread, spray 
        "starts_thr": int(w.startswith("thr")),  # three, through, throw 
        "suffix_ful": int(w.endswith("ful")),  # beautiful, wonderful 
        "suffix_less": int(w.endswith("less")),  # endless, helpless 
        "suffix_ment": int(w.endswith("ment")),  # movement, government
        "contains_following": int(bool(re.search(r'[vwjxzfcq]', w))), #uncommon letters in filipino
        
        # OTH features (symbols, numbers, abbreviations, onomatopoeia, unknown)
        "is_very_short": int(len(w) <= 2),  # "ok", ".", ",", "lol"
        "is_single_char": int(len(w) == 1),  # Single punctuation or letter
        "has_digit": int(any(c.isdigit() for c in word)),  # Numbers: "123", "1st"
        "has_symbol": int(bool(re.search(r'[^A-Za-z0-9ñÑ]', word))),  # Punctuation: ".", "!", "?"
        "is_all_caps": int(word.isupper() and len(word) > 1),  # Abbreviations: "USA", "LOL", "OMG"
        "is_internet_slang": int(w in ["lol", "omg", "wtf", "brb", "idk", "tbh", "imo", "btw", "fyi", "smh"]),  # Common internet slang
        "is_laugh_pattern": int(bool(re.search(r'^(ha|he|hi|ho|hu)\1+$', w))),  # haha, hehe, hihi, hoho, huhu
        "is_repeated_syllable": int(bool(re.search(r'^([a-z]{2})\1+$', w)) and len(w) == 4),  # haha, huhu, hihi (exactly 4 chars)
        "is_mixed_case": int(any(c.isupper() for c in word[1:]) and any(c.islower() for c in word)),  # CamelCase, weird caps
        "has_repeated_chars": int(bool(re.search(r'(.)\1{2,}', w))),  # Onomatopoeia: "hahaha", "zzz", "awww"
        "all_consonants": int(len(w) > 0 and vowel_count == 0 and w.isalpha()),  # "psst", "shh", "brr"
        "vowel_consonant_ratio": vowel_count / max(1, consonant_count),  # Unusual ratios
        "has_numbers_and_letters": int(any(c.isdigit() for c in word) and any(c.isalpha() for c in word)),  # "covid19", "21st"
        "is_emoticon": int(bool(re.search(r'[:\-;][)(DPO]|[)(DPO][:;]', word))),  # :), :D, :(, ;)
        "has_multiple_punctuation": int(len(re.findall(r'[^\w\s]', word)) > 1),  # "!!!", "?!?", "..."
        "is_pure_punctuation": int(not any(c.isalnum() for c in word) and len(word) > 0),  # ".", "...", "!!!"
        "uncommon_letter_combo": int(bool(re.search(r'[qxz]{2}|[bcdfghjklmnpqrstvwxyz]{4,}', w))),  # Weird combos
    }


def prepare_dataset(df):
    #this onvert dataset of words and labels into features and targets
    # Convert WORDS into features (X)
    X = pd.DataFrame([extract_features(w) for w in df["word"]])
    # "gusto" -> {length: 5, suffix_o: 1, ...}  -> features
    
    # Keep labels as targets (y)
    y = df["label"]
    # "FIL" stays as "FIL"  ← TARGET
    
    return X, y

#reading the training data (csv file)
data = pd.read_csv("final_annotations.csv")
X, y = prepare_dataset(data)

# 70-15-15 train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# HYPERPARAMETER TUNING (Optuna GridSearch)
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - Testing 72 combinations")
print("="*60)

# Search space: 2×4×3×3 = 72 combinations
search_space = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, 20, 25],
    'min_samples_leaf': [3, 5, 10],
    'min_samples_split': [2, 5, 10, 15]
}

def objective(trial):
    #train model with trial parameters and return validation accuracy
    model = DecisionTreeClassifier(
        criterion=trial.suggest_categorical('criterion', search_space['criterion']),
        max_depth=trial.suggest_categorical('max_depth', search_space['max_depth']),
        min_samples_leaf=trial.suggest_categorical('min_samples_leaf', search_space['min_samples_leaf']),
        min_samples_split=trial.suggest_categorical('min_samples_split', search_space['min_samples_split']),
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return float(model.score(X_val, y_val))

# Run grid search
study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space, seed=42))
study.optimize(objective, show_progress_bar=True)

# Display results
print(f"\n{'='*60}")
print(f"BEST PARAMETERS: {study.best_params}")
print(f"Best Validation Accuracy: {study.best_value:.4f}")
print(f"{'='*60}\n")

# Train final model with best parameters
model = DecisionTreeClassifier(**study.best_params, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# accuracy checking (to be deleted when submitting)
y_pred = model.predict(X_test)
print("Decision Tree Evaluation")
print("---------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "pinoybot_model.pkl")
print("\n✅ Model saved as pinoybot_model.pkl") #could delete this before submission (debatable)