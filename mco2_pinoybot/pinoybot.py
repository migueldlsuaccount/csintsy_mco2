"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import re
from typing import List
import pandas as pd
import joblib

# Load the trained model 
model = joblib.load("pinoybot_model.pkl")

  # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions

# Helper function to extract features from a word
def extract_features(word):
    #extract handcrafted features from a given word
    if not isinstance(word, str):
        word = str(word) if word is not None else ""
    
    w = word.lower()
    
    # pre-compile regex patterns
    consonant_pattern = r'[bcdfghjklmnpqrstvwxyz]'
    consonant_clusters = len(re.findall(consonant_pattern + r'{2,}', w))
    starts_consonant_cluster = int(bool(re.match(consonant_pattern + r'{2,}', w)))
    
    # calculate vowel ratio
    vowel_count = sum(c in 'aeiou' for c in w)
    consonant_count = sum(c in 'bcdfghjklmnpqrstvwxyz' for c in w)
    
    return {
        "length": len(w),
        "vowel_ratio": vowel_count / max(1, len(w)),
        "consonant_clusters": consonant_clusters,
        "starts_consonant_cluster": starts_consonant_cluster,
        
        # FIL features
        "has_ng": int("ng" in w), # common in Filipino: "ng", "mang", "sang"
        "has_mga": int("mga" in w), # common in Filipino: "mga"
        "prefix_mag": int(w.startswith("mag")), # common in Filipino: "mag"
        "prefix_nag": int(w.startswith("nag")), # common in Filipino: "nag"
        "prefix_pag": int(w.startswith("pag")), # common in Filipino: "pag"
        "prefix_ka": int(w.startswith("ka")), # common in Filipino: "ka"
        "suffix_an": int(w.endswith("an")), # common in Filipino: "an"
        "suffix_in": int(w.endswith("in")), # common in Filipino: "in"
        "suffix_han": int(w.endswith("han")), # common in Filipino: "han"
        "suffix_o": int(w.endswith("o") and len(w) > 2), # common in Filipino: gusto, pero, ako, ko, ito, tao
        "has_uso": int("uso" in w), # common in Filipino: puso, kapuso, dugo, grupo
        "has_sto": int("sto" in w), # common in Filipino borrowed words: gusto, wasto, Agosto, gastos
        "starts_gu": int(w.startswith("gu")), # common Filipino verbs: gusto, gumawa, gumagawa, gumamit, gumising
        "pattern_cuco": int(bool(re.search(r'[bcdfghjklmnpqrstvwxyz]u[bcdfghjklmnpqrstvwxyz]o$', w))), # Filipino pattern: gusto, puso, turo, luto
        
        # ENG features
        "suffix_ing": int(w.endswith("ing")), # learning, programming, procastinating LOOOOLLL
        "suffix_ed": int(w.endswith("ed")), #cooked, depressed, mega-cooked...
        "suffix_ly": int(w.endswith("ly")), # lovely, quickly, happily
        "suffix_tion": int(w.endswith("tion")), # nation, station, relation
        "suffix_ness": int(w.endswith("ness")), # happiness, sadness, kindness
        "suffix_er": int(w.endswith("er")), # teacher, player, runner, nigger
        "suffix_est": int(w.endswith("est")), # biggest, fastest, strongest
        "suffix_s": int(w.endswith("s") and len(w) > 2),  # English plurals
        "suffix_y": int(w.endswith("y") and len(w) > 2),  # today, yesterday, happy
        "prefix_un": int(w.startswith("un")), # unhappy, undone
        "prefix_re": int(w.startswith("re")), # redo, rewrite, reread
        "prefix_pre": int(w.startswith("pre")), # preview, pretest, pre-enlist
        "prefix_dis": int(w.startswith("dis")), # disconnect, disapprove
        "has_ing": int(w.endswith("ing")), # learning, programming, procrastinating ??? meron nang suffix_ing
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
        "has_ea": int("ea" in w),  # weather, beautiful, eat
        "has_ee": int("ee" in w),  # see, tree, meet, feel (NEW - safe English marker)
        "has_ou": int("ou" in w),  # house, mouse, out, about (NEW - safe English marker)
        "has_ai": int("ai" in w),  # rain, wait, main (NEW - rare in Filipino)
        "starts_str": int(w.startswith("str")),  # street, string, strong (NEW - never in native Filipino)
        "starts_spr": int(w.startswith("spr")),  # spring, spread, spray (NEW - never in native Filipino)
        "starts_thr": int(w.startswith("thr")),  # three, through, throw (NEW - never in native Filipino)
        "suffix_ful": int(w.endswith("ful")),  # beautiful, wonderful (NEW - pure English suffix)
        "suffix_less": int(w.endswith("less")),  # endless, helpless (NEW - pure English suffix)
        "suffix_ment": int(w.endswith("ment")),  # movement, government (NEW - pure English suffix)
        "contains_v": int("v" in w), # Filipino rarely uses 'v'
        "contains_w": int("w" in w), # Filipino rarely uses 'w'
        "contains_j": int("j" in w), # Filipino rarely uses 'j'
        "contains_x": int("x" in w), # Filipino rarely uses 'x'
        "contains_z": int("z" in w), # Filipino rarely uses 'z'
        "contains_f": int("f" in w),  # Filipino rarely uses 'f'
        
        # OTH features (symbols, numbers, abbreviations, onomatopoeia, unknown)
        "is_very_short": int(len(w) <= 2),  # Abbreviations/symbols: "ok", ".", ",", "lol"
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

# MAIN FUNCTION
def tag_language(tokens: List[str]) -> List[str]:
    """
    Takes a list of tokens (words) and returns a list of language tags.
    
    Args:
        tokens: List of word tokens (strings).
    
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    
    Example:
        >>> tag_language(["Love", "kita", ".", "nagpost", "ako", "today"])
        ['ENG', 'FIL', 'OTH', 'FIL', 'FIL', 'ENG']
    """
    features = [extract_features(word) for word in tokens]
    X = pd.DataFrame(features)
    preds = model.predict(X)
    return preds.tolist()

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:", tags)