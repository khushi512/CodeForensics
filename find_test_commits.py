import pandas as pd

df = pd.read_csv('data/datasets/training_dataset.csv')

# Find commits that were labeled as bug-inducing
risky = df[df['is_bug_inducing'] == True].head(5)

print("="*60)
print("HIGH-RISK COMMITS TO TEST (from training data)")
print("="*60)

for idx, row in risky.iterrows():
    print(f"\nCommit: {row['sha'][:12]}...")
    print(f"  Bug-inducing: Yes")
    print(f"  Complexity: {row['complexity_delta']}")
    print(f"  Files: {row['files_changed']}")
    print(f"  Full SHA: {row['sha']}")

print("\n" + "="*60)
print("SAFE COMMITS (not bug-inducing)")
print("="*60)

safe = df[df['is_bug_inducing'] == False].head(3)
for idx, row in safe.iterrows():
    print(f"\nCommit: {row['sha'][:12]}...")
    print(f"  Bug-inducing: No")
    print(f"  Full SHA: {row['sha']}")
