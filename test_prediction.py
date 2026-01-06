from src.model.predict import predict_risk

# Test prediction with sample commit features
test_features = {
    'hour_of_day': 23,  # Late night
    'day_of_week': 4,   # Friday
    'is_weekend': False,
    'files_changed': 10,
    'py_files_modified': 3,
    'lines_added': 200,
    'lines_deleted': 50,
    'net_lines': 150,
    'complexity_delta': 5,
    'avg_file_churn': 30
}

result = predict_risk(test_features)
print("="*50)
print("CODEFORENSICS - Test Prediction")
print("="*50)
print(f"Risk Score: {result['risk_score']:.1%}")
print(f"Risk Level: {result['risk_level']}")
print("\nFeatures used:")
for k, v in test_features.items():
    print(f"  {k:20} = {v}")
