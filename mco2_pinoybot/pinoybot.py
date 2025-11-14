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
    
    has_repeating = 0
    for i in range(len(w) - 3):
        if w[i] in 'bdghjklmnprstwy' and w[i+1] in 'aeiou':
            if w[i:i+2] == w[i+2:i+4]:
                has_repeating = 1
                break
    
    return {
        "length": len(w),
        "vowel_ratio": vowel_count / max(1, len(w)),
        "consonant_clusters": consonant_clusters,
        "starts_consonant_cluster": starts_consonant_cluster,
        
        # FIL features
        "has_reduplication": int(has_repeating), #repeating, "gaga(wa)", "(ma)nana(lo)"
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
        "suffix_mes": int(w.endswith("mes")), #games, frames, names...
        "suffix_ine": int(w.endswith("ine")), #Dopamine, Creatine, Valentine...
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


# MAIN FUNCTION
def tag_language(df: pd.DataFrame, model) -> List[str]:
    """
    Takes a DataFrame with 'word' column and returns a list of language tags.
    """
    if df.empty:
        return []
    
    X = prepare_features(df)
    preds = model.predict(X)
    return preds.tolist()

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataset of words into features.
    """
    X = pd.DataFrame([extract_features(w) for w in df["word"]])
    return X

import pandas as pd

def read_tokens_from_file(filepath: str) -> pd.DataFrame:
    """
    Reads CSV, TXT, or XLSX and extracts the 'word' column into a DataFrame.
    Works with headers. Includes all characters in the word.
    
    Args:
        filepath: Path to the input file
    
    Returns:
        DataFrame with a single 'word' column
    """
    words = []

    if filepath.lower().endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        # TXT file, one token per line
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words.append(line)
        return pd.DataFrame({'word': words})

    # Automatically detect 'word' column
    if "word" in df.columns:
        words = df["word"].astype(str).tolist()
    else:
        # Fallback: take the 3rd column (index 2)
        words = df.iloc[:, 2].astype(str).tolist()

    return pd.DataFrame({'word': words})


def write_tags_to_file(tags: List[str], output_filepath: str):
    """
    Write predicted tags to a file, one tag per line.
    """
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tags))


if __name__ == "__main__":

    print("=" * 50)
    print("Language Tagger - Code-Switched Text")
    print("=" * 50)

    input_file = input("\nEnter input filename (txt/csv/xlsx): ").strip()
    output_file = input("Enter output filename (default: output_tags.txt): ").strip()
    if not output_file:
        output_file = "output_tags.txt"

    try:
        # Read file
        print(f"\nReading tokens from: {input_file}")
        df = read_tokens_from_file(input_file)
        print(f"Found {len(df)} tokens")
        print("\nFirst 5 words:")
        print(df.head(100))

        # Load trained model
        model = joblib.load('pinoybot_model.pkl')
        print("\nModel loaded successfully!")

        # Predict tags
        print("\nPredicting tags...")
        tags = tag_language(df, model)

        print("\nFirst 5 predictions:")
        for i in range(min(100, len(tags))):
            print(f"  {df.iloc[i]['word']} -> {tags[i]}")

        # Save output
        write_tags_to_file(tags, output_file)
        print(f"\n✓ Tags successfully written to: {output_file}")
        print(f"  Total tokens processed: {len(tags)}")

    except FileNotFoundError:
        print(f"\n✗ Error: File '{input_file}' not found!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")