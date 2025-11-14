"""
100 Non-Biased Test Cases for PinoyBot
=======================================
Realistic code-switched Filipino-English scenarios with balanced distribution.
Includes common words, edge cases, and real-world usage patterns.
"""
import pandas as pd
from train_model import extract_features
import joblib

# Load model
model = joblib.load('pinoybot_model.pkl')

# 100 realistic test cases from code-switched Filipino contexts
# Distribution: ~40 FIL, ~35 ENG, ~25 OTH (realistic imbalance)
test_cases = [ ("Kumain", "FIL"), ("ka", "FIL"), ("na", "FIL"), ("ba", "FIL"), ("today", "ENG"), ("?", "OTH"), ("I", "ENG"), ("love", "ENG"), ("eating", "ENG"), ("sinigang", "FIL"), ("sa", "FIL"), ("gabi", "FIL"), (".", "OTH"), ("Pupunta", "FIL"), ("ako", "FIL"), ("sa", "FIL"), ("mall", "ENG"), ("later", "ENG"), ("para", "FIL"), ("bumili", "FIL"), ("ng", "FIL"), ("new", "ENG"), ("shoes", "ENG"), (".", "OTH"), ("Grabe", "FIL"), ("yung", "FIL"), ("traffic", "ENG"), ("kanina", "FIL"), (",", "OTH"), ("sobrang", "FIL"), ("late", "ENG"), ("na", "FIL"), ("ako", "FIL"), ("!", "OTH"), ("Can", "ENG"), ("you", "ENG"), ("help", "ENG"), ("me", "ENG"), ("mag-review", "FIL"), ("para", "FIL"), ("sa", "FIL"), ("exam", "ENG"), ("bukas", "FIL"), ("?", "OTH"),("Ang", "FIL"), ("ganda", "FIL"), ("ng", "FIL"), ("weather", "ENG"), ("ngayon", "FIL"), (",", "OTH"), ("perfect", "ENG"), ("para", "FIL"), ("sa", "FIL"), ("beach", "ENG"), (".", "OTH"), ("Nag-order", "FIL"), ("na", "FIL"), ("ako", "FIL"), ("ng", "FIL"), ("pizza", "ENG"), ("online", "ENG"), (",", "OTH"), ("darating", "FIL"), ("in", "ENG"), ("30", "OTH"), ("minutes", "ENG"), (".", "OTH"), ("Sabi", "FIL"), ("niya", "FIL"), ("meeting", "ENG"), ("daw", "FIL"), ("later", "ENG"), ("at", "FIL"), ("5pm", "OTH"), (".", "OTH"), ("Nagluto", "FIL"), ("ako", "FIL"), ("ng", "FIL"), ("adobo", "FIL"), ("and", "ENG"), ("fried", "ENG"), ("rice", "ENG"), ("for", "ENG"), ("dinner", "ENG"), (".", "OTH"), ("Ang", "FIL"), ("init", "FIL"), ("naman", "FIL"), ("dito", "FIL"), (",", "OTH"), ("can", "ENG"), ("we", "ENG"), ("turn", "ENG"), ("on", "ENG"), ("the", "ENG"), ("aircon", "ENG"), ("?", "OTH"),  ("May", "FIL"), ("presentation", "ENG"), ("pa", "FIL"), ("ako", "FIL"), ("bukas", "FIL"), (",", "OTH"), ("need", "ENG"), ("ko", "FIL"), ("mag-prepare", "FIL"), (".", "OTH"), ("Deadline", "ENG"), ("na", "FIL"), ("namin", "FIL"), ("sa", "FIL"), ("project", "ENG"), ("this", "ENG"), ("Friday", "ENG"), ("!", "OTH"), ("Teacher", "ENG"), ("namin", "FIL"), ("is", "ENG"), ("so", "ENG"), ("strict", "ENG"), (",", "OTH"), ("grabe", "FIL"), ("yung", "FIL"), ("requirements", "ENG"), (".", "OTH"), ("Nag-apply", "FIL"), ("na", "FIL"), ("ako", "FIL"), ("sa", "FIL"), ("job", "ENG"), ("posting", "ENG"), ("na", "FIL"), ("yun", "FIL"), (".", "OTH"), ("Kailangan", "FIL"), ("ko", "FIL"), ("mag-study", "FIL"), ("para", "FIL"), ("sa", "FIL"), ("finals", "ENG"), ("next", "ENG"), ("week", "ENG"), (".", "OTH"), ("My", "ENG"), ("boss", "ENG"), ("is", "ENG"), ("super", "ENG"), ("bait", "FIL"), (",", "OTH"), ("lagi", "FIL"), ("niya", "FIL"), ("ako", "FIL"), ("binibigyan", "FIL"), ("ng", "FIL"), ("tips", "ENG"), (".", "OTH"), ("Nakapag-submit", "FIL"), ("ka", "FIL"), ("na", "FIL"), ("ba", "FIL"), ("ng", "FIL"), ("requirements", "ENG"), ("mo", "FIL"), ("?", "OTH"), ("The", "ENG"), ("conference", "ENG"), ("was", "ENG"), ("so", "ENG"), ("boring", "ENG"), (",", "OTH"), ("parang", "FIL"), ("matutulog", "FIL"), ("na", "FIL"), ("ako", "FIL"), ("lol", "OTH"), (".", "OTH"), ("Nag-overtime", "FIL"), ("ako", "FIL"), ("kagabi", "FIL"), ("para", "FIL"), ("matapos", "FIL"), ("yung", "FIL"), ("report", "ENG"), (".", "OTH"), ("Saan", "FIL"), ("ba", "FIL"), ("yung", "FIL"), ("office", "ENG"), ("nila", "FIL"), ("?", "OTH"), ("I", "ENG"), ("think", "ENG"), ("sa", "FIL"), ("Makati", "FIL"), (".", "OTH"),  ("May", "FIL"), ("nag-post", "FIL"), ("na", "FIL"), ("naman", "FIL"), ("ng", "FIL"), ("selfie", "ENG"), (",", "OTH"), ("hahaha", "OTH"), ("!", "OTH"), ("Grabe", "FIL"), ("yung", "FIL"), ("comments", "ENG"), ("sa", "FIL"), ("Facebook", "ENG"), (",", "OTH"), ("puro", "FIL"), ("fake", "ENG"), ("news", "ENG"), (".", "OTH"), ("Download", "ENG"), ("mo", "FIL"), ("yung", "FIL"), ("app", "ENG"), ("na", "FIL"), ("yun", "FIL"), (",", "OTH"), ("useful", "ENG"), ("siya", "FIL"), ("!", "OTH"), ("Nag-viral", "FIL"), ("yung", "FIL"), ("video", "ENG"), ("natin", "FIL"), (",", "OTH"), ("omg", "OTH"), ("100k", "OTH"), ("views", "ENG"), ("na", "FIL"), ("!", "OTH"), ("My", "ENG"), ("phone", "ENG"), ("is", "ENG"), ("low", "ENG"), ("bat", "ENG"), (",", "OTH"), ("pwede", "FIL"), ("ba", "FIL"), ("makahiram", "FIL"), ("ng", "FIL"), ("charger", "ENG"), ("?", "OTH"), ("Nag-update", "FIL"), ("na", "FIL"), ("ako", "FIL"), ("ng", "FIL"), ("status", "ENG"), ("ko", "FIL"), ("sa", "FIL"), ("Instagram", "ENG"), (".", "OTH"), ("Check", "ENG"), ("mo", "FIL"), ("yung", "FIL"), ("link", "ENG"), ("na", "FIL"), ("send", "ENG"), ("ko", "FIL"), ("sa", "FIL"), ("chat", "ENG"), (".", "OTH"), ("Nawala", "FIL"), ("yung", "FIL"), ("WiFi", "ENG"), ("password", "ENG"), (",", "OTH"), ("ano", "FIL"), ("ba", "FIL"), ("ulit", "FIL"), ("?", "OTH"), ("Screenshot", "ENG"), ("mo", "FIL"), ("tapos", "FIL"), ("i-send", "FIL"), ("mo", "FIL"), ("sa", "FIL"), ("akin", "FIL"), (".", "OTH"), ("Ang", "FIL"), ("bagal", "FIL"), ("ng", "FIL"), ("internet", "ENG"), ("dito", "FIL"), (",", "OTH"), ("di", "FIL"), ("ako", "FIL"), ("maka-load", "FIL"), (".", "OTH"), ("Tara", "FIL"), ("na", "FIL"), ("sa", "FIL"), ("grocery", "ENG"), (",", "OTH"), ("bili", "FIL"), ("tayo", "FIL"), ("ng", "FIL"), ("ingredients", "ENG"), (".", "OTH"), ("How", "ENG"), ("much", "ENG"), ("ba", "FIL"), ("yung", "FIL"), ("bag", "ENG"), ("na", "FIL"), ("yan", "FIL"), ("?", "OTH"), ("Nag-sale", "FIL"), ("daw", "FIL"), ("sa", "FIL"), ("SM", "OTH"), (",", "OTH"), ("up", "ENG"), ("to", "ENG"), ("50%", "OTH"), ("off", "ENG"), ("!", "OTH"), ("Masarap", "FIL"), ("ba", "FIL"), ("yung", "FIL"), ("restaurant", "ENG"), ("na", "FIL"), ("yan", "FIL"), ("?", "OTH"), ("I", "ENG"), ("want", "ENG"), ("to", "ENG"), ("try", "ENG"), (".", "OTH"), ("Order", "ENG"), ("na", "FIL"), ("tayo", "FIL"), ("ng", "FIL"), ("milk", "ENG"), ("tea", "ENG"), (",", "OTH"), ("gusto", "FIL"), ("ko", "FIL"), ("yung", "FIL"), ("brown", "ENG"), ("sugar", "ENG"), (".", "OTH"), ("Ang", "FIL"), ("mahal", "FIL"), ("naman", "FIL"), ("ng", "FIL"), ("presyo", "FIL"), ("dito", "FIL"), (",", "OTH"), ("lets", "ENG"), ("go", "ENG"), ("somewhere", "ENG"), ("else", "ENG"), (".", "OTH"), ("May", "FIL"), ("discount", "ENG"), ("ba", "FIL"), ("if", "ENG"), ("cash", "ENG"), ("payment", "ENG"), ("?", "OTH"), ("Bili", "FIL"), ("ka", "FIL"), ("na", "FIL"), ("ng", "FIL"), ("snacks", "ENG"), ("para", "FIL"), ("sa", "FIL"), ("movie", "ENG"), ("night", "ENG"), (".", "OTH"), ("Naubos", "FIL"), ("na", "FIL"), ("yung", "FIL"), ("stock", "ENG"), ("ng", "FIL"), ("yung", "FIL"), ("bestseller", "ENG"), ("nila", "FIL"), (":(", "OTH"), ("Free", "ENG"), ("delivery", "ENG"), ("daw", "FIL"), ("kapag", "FIL"), ("minimum", "ENG"), ("of", "ENG"), ("500", "OTH"), ("pesos", "FIL"), (".", "OTH"),  ("Saan", "FIL"), ("tayo", "FIL"), ("pupunta", "FIL"), ("hello", "ENG"), ("weekend", "ENG"), ("?", "OTH"), ("Nag-book", "FIL"), ("na", "FIL"), ("ako", "FIL"), ("ng", "FIL"), ("hotel", "ENG"), ("sa", "FIL"), ("Boracay", "FIL"), ("for", "ENG"), ("next", "ENG"), ("month", "ENG"), (".", "OTH"), ("The", "ENG"), ("flight", "ENG"), ("was", "ENG"), ("delayed", "ENG"), (",", "OTH"), ("sobrang", "FIL"), ("tagal", "FIL"), ("namin", "FIL"), ("naghintay", "FIL"), (".", "OTH"), ("Tara", "FIL"), ("road", "ENG"), ("trip", "ENG"), ("tayo", "FIL"), ("papuntang", "FIL"), ("Baguio", "FIL"), ("!", "OTH"), ("Nag-enjoy", "FIL"), ("talaga", "FIL"), ("ako", "FIL"), ("sa", "FIL"), ("vacation", "ENG"), (",", "OTH"), ("ayoko", "FIL"), ("pang", "FIL"), ("umuwi", "FIL"), ("haha", "OTH"), (".", "OTH"), ("May", "FIL"), ("pasalubong", "FIL"), ("ka", "FIL"), ("ba", "FIL"), ("from", "ENG"), ("your", "ENG"), ("trip", "ENG"), ("?", "OTH"), ("Nakita", "FIL"), ("mo", "FIL"), ("ba", "FIL"), ("yung", "FIL"), ("sunset", "ENG"), ("?", "OTH"), ("Grabe", "FIL"), (",", "OTH"), ("so", "ENG"), ("beautiful", "ENG"), ("!", "OTH"), ("Magkano", "FIL"), ("ba", "FIL"), ("yung", "FIL"), ("ticket", "ENG"), ("papuntang", "FIL"), ("Cebu", "FIL"), ("?", "OTH"), ("I", "ENG"), ("cant", "ENG"), ("wait", "ENG"), ("for", "ENG"), ("our", "ENG"), ("beach", "ENG"), ("trip", "ENG"), ("bukas", "FIL"), ("!", "OTH"), ("!", "OTH"), ("!", "OTH"), ("Nag-enjoy", "FIL"), ("ba", "FIL"), ("kayo", "FIL"), ("sa", "FIL"), ("theme", "ENG"), ("park", "ENG"), ("?", "OTH"), ("Bet", "ENG"), ("ko", "FIL"), ("yung", "FIL"), ("roller", "ENG"), ("coaster", "ENG"), ("!", "OTH"), 
]# Verify we have 100 cases
assert len(test_cases) == 487, f"Expected 100 cases, got {len(test_cases)}"

# Count distribution
fil_count = sum(1 for _, label in test_cases if label == "FIL")
eng_count = sum(1 for _, label in test_cases if label == "ENG")
oth_count = sum(1 for _, label in test_cases if label == "OTH")

print("=" * 80)
print("100 NON-BIASED TEST CASES FOR PINOYBOT")
print("=" * 80)
print(f"\nTest Distribution:")
print(f"  Filipino (FIL): {fil_count} cases ({fil_count}%)")
print(f"  English (ENG):  {eng_count} cases ({eng_count}%)")
print(f"  Other (OTH):    {oth_count} cases ({oth_count}%)")
print(f"  TOTAL:          {len(test_cases)} cases")
print("\n" + "=" * 80)

# Track results by category
results = {"FIL": {"correct": 0, "total": 0}, 
           "ENG": {"correct": 0, "total": 0}, 
           "OTH": {"correct": 0, "total": 0}}
errors = []

# Run predictions
for word, expected in test_cases:
    features = extract_features(word)
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    
    results[expected]["total"] += 1
    
    if prediction == expected:
        status = "✅"
        results[expected]["correct"] += 1
    else:
        status = "❌"
        errors.append((word, expected, prediction))
    
    print(f"{status} {word:20s} -> {prediction:3s} (expected: {expected})")

# Calculate accuracy metrics
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

overall_correct = sum(results[cat]["correct"] for cat in results)
overall_total = sum(results[cat]["total"] for cat in results)
overall_accuracy = 100 * overall_correct / overall_total

print(f"\nOverall Accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.2f}%")
print("\nPer-Class Performance:")

for category in ["FIL", "ENG", "OTH"]:
    correct = results[category]["correct"]
    total = results[category]["total"]
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"  {category}: {correct}/{total} = {accuracy:.2f}%")

# Show errors
if errors:
    print("\n" + "=" * 80)
    print(f"ERRORS ({len(errors)} total):")
    print("=" * 80)
    for word, expected, predicted in errors:
        print(f"  '{word}' -> predicted {predicted} (should be {expected})")
else:
    print("\n🎉 Perfect score! No errors!")

# Calculate confusion matrix manually
confusion = {
    "FIL": {"FIL": 0, "ENG": 0, "OTH": 0},
    "ENG": {"FIL": 0, "ENG": 0, "OTH": 0},
    "OTH": {"FIL": 0, "ENG": 0, "OTH": 0}
}

for word, expected in test_cases:
    features = extract_features(word)
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    confusion[expected][prediction] += 1

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
print("\n         Predicted")
print("Actual   FIL  ENG  OTH")
print("-" * 30)
for actual in ["FIL", "ENG", "OTH"]:
    counts = [confusion[actual][pred] for pred in ["FIL", "ENG", "OTH"]]
    print(f"{actual:6s}   {counts[0]:3d}  {counts[1]:3d}  {counts[2]:3d}")

print("\n" + "=" * 80)
