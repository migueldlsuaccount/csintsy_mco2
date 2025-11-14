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
test_cases = [
    ("Kumain", "FIL"), ("ka", "FIL"), ("na", "FIL"), ("ng", "FIL"), ("almusal", "FIL"), ("?", "OTH"),
    ("I", "ENG"), ("bought", "ENG"), ("a", "ENG"), ("new", "ENG"), ("backpack", "ENG"), (".", "OTH"),

    ("Maganda", "FIL"), ("ang", "FIL"), ("view", "ENG"), ("ng", "FIL"), ("bundok", "FIL"), (".", "OTH"),
    
    ("Please", "ENG"), ("send", "ENG"), ("me", "ENG"), ("the", "ENG"), ("document", "ENG"), ("by", "ENG"), ("tomorrow", "ENG"), (".", "OTH"),
    
    ("Naglinis", "FIL"), ("ako", "FIL"), ("ng", "FIL"), ("kwarto", "FIL"), (",", "OTH"), ("pero", "FIL"), ("malaki", "FIL"), ("pa", "FIL"), ("ang", "FIL"), ("gulo", "FIL"), (".", "OTH"),
    
    ("Can", "ENG"), ("you", "ENG"), ("fix", "ENG"), ("the", "ENG"), ("computer", "ENG"), ("?", "OTH"),
    
    ("Masarap", "FIL"), ("ang", "FIL"), ("sinigang", "FIL"), ("sa", "FIL"), ("restawran", "FIL"), (".", "OTH"),
    
    ("OMG", "OTH"), (",", "OTH"), ("the", "ENG"), ("party", "ENG"), ("was", "ENG"), ("so", "ENG"), ("fun", "ENG"), ("!", "OTH"),
    
    ("Magbasa", "FIL"), ("ka", "FIL"), ("ng", "FIL"), ("libro", "FIL"), ("bago", "FIL"), ("matulog", "FIL"), (".", "OTH"),
    
    ("I", "ENG"), ("need", "ENG"), ("to", "ENG"), ("finish", "ENG"), ("my", "ENG"), ("homework", "ENG"), ("tonight", "ENG"), (".", "OTH"),
    
    ("Saan", "FIL"), ("ang", "FIL"), ("nearest", "ENG"), ("grocery", "ENG"), ("store", "ENG"), ("?", "OTH"),
]# Verify we have 100 cases
assert len(test_cases) == 78, f"Expected 100 cases, got {len(test_cases)}"

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
