"""
Verification & Test Suite for SMS Spam Message Detection.
Team 2 - Natural Language Processing

Tests the 8 mandatory scenarios required by Section 10:
1. Normal personal message
2. Promotional SMS
3. Message containing a suspicious link
4. Message mentioning a prize or reward
5. Ordinary bank notification
6. Empty input
7. Message with numbers and punctuation
8. Message containing both spam-like and legitimate language
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_message

TEST_CASES = [
    {
        "id": "TC-01",
        "category": "Normal Personal Message",
        "text": "Hey Mom, I am on my way home from college. Do we need anything from the grocery store?",
        "intent": "Legitimate communication between family members."
    },
    {
        "id": "TC-02",
        "category": "Promotional SMS",
        "text": "EXCLUSIVE SALE! Get 50% discount on all designer jackets this weekend only. Shop now at your nearest store or call 0800-SALE-NOW.",
        "intent": "Unsolicited promotional advertising with call to action."
    },
    {
        "id": "TC-03",
        "category": "Suspicious Link Message",
        "text": "Urgent alert: Your postal package delivery could not be completed. Update your delivery address now at http://parcel-redelivery-secure.net/track?id=99281",
        "intent": "Phishing lure with suspicious external link."
    },
    {
        "id": "TC-04",
        "category": "Prize / Reward SMS",
        "text": "Congratulations! You have been selected as our £1,000 weekly cash prize winner! Text CLAIM to 87012 or call 09061743811 to redeem.",
        "intent": "Classic lottery/cash reward lure with premium rate numbers."
    },
    {
        "id": "TC-05",
        "category": "Ordinary Bank Notification",
        "text": "Your checking account ending in 8419 was debited for $24.50 at Corner Cafe on 10/14. Available balance: $1,280.12. If unauthorized, reply HELP.",
        "intent": "Standard transactional alert from a financial institution."
    },
    {
        "id": "TC-06",
        "category": "Empty Input",
        "text": "    ",
        "intent": "Edge case validation for empty or whitespace-only input."
    },
    {
        "id": "TC-07",
        "category": "Numbers & Punctuation",
        "text": "Meeting room #402, 10:30 AM tomorrow -- please bring 3 printed copies of doc #712! Thanks :)",
        "intent": "Legitimate work message with high density of punctuation and numbers."
    },
    {
        "id": "TC-08",
        "category": "Mixed Spam & Legitimate Language",
        "text": "Hey John, did you hear about that free pizza offer the university cafeteria was talking about yesterday? Sounds cool!",
        "intent": "Legitimate informal chat that mentions promotional keywords like 'free offer'."
    }
]


def run_all_tests():
    print("=" * 80)
    print(" SMS SPAM DETECTION - TEST SCENARIO EXECUTION")
    print(" Team 2 | Natural Language Processing")
    print("=" * 80)

    results = []
    
    for case in TEST_CASES:
        print(f"\n[{case['id']}] Category: {case['category']}")
        print(f" Input Text: \"{case['text']}\"")
        res = predict_message(case["text"])
        
        status = res.get("status")
        pred = res.get("prediction")
        conf = res.get("confidence", 0.0)
        spam_prob = res.get("spam_probability", 0.0)
        ham_prob = res.get("ham_probability", 0.0)
        explanation = res.get("explanation", res.get("error_message", ""))
        
        print(f" -> Output: {pred} | Confidence: {conf * 100:.2f}% (Spam Prob: {spam_prob*100:.2f}%, Ham Prob: {ham_prob*100:.2f}%)")
        print(f" -> Explanation: {explanation}")
        
        results.append({
            "test_id": case["id"],
            "category": case["category"],
            "input_text": case["text"],
            "prediction": pred,
            "confidence_pct": round(conf * 100, 2),
            "spam_prob_pct": round(spam_prob * 100, 2),
            "ham_prob_pct": round(ham_prob * 100, 2),
            "explanation": explanation
        })
        
    # Save test results for documentation and report
    output_path = os.path.join(PROJECT_ROOT, "models", "test_scenarios_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Test Suite] Saved test scenario results to: {output_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_all_tests()
