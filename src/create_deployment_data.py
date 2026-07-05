import os
import pandas as pd


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data",
)

SOURCE_PATH = os.path.join(
    DATA_DIR,
    "creditcard.csv",
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "fraudshield_transactions.csv",
)


print("=" * 65)
print("FRAUDSHIELD AI - DEPLOYMENT DATASET GENERATOR")
print("=" * 65)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n[1/5] Loading original transaction dataset...")

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(
        f"Dataset was not found: {SOURCE_PATH}"
    )

data = pd.read_csv(SOURCE_PATH)

print(f"Transactions loaded : {len(data):,}")
print(f"Features            : {len(data.columns)}")


# ============================================================
# VALIDATE TARGET
# ============================================================

print("\n[2/5] Validating fraud target...")

if "Class" not in data.columns:
    raise ValueError(
        "The dataset must contain the Class column."
    )

legitimate_data = data[
    data["Class"] == 0
].copy()

fraud_data = data[
    data["Class"] == 1
].copy()

print(
    f"Legitimate transactions : "
    f"{len(legitimate_data):,}"
)

print(
    f"Fraud transactions      : "
    f"{len(fraud_data):,}"
)


# ============================================================
# CREATE REPRESENTATIVE DEPLOYMENT POPULATION
# ============================================================

print(
    "\n[3/5] Creating deployment transaction population..."
)

LEGITIMATE_SAMPLE_SIZE = 30000

legitimate_sample = legitimate_data.sample(
    n=min(
        LEGITIMATE_SAMPLE_SIZE,
        len(legitimate_data),
    ),
    random_state=42,
)

# Keep every known fraud example so the deployed dashboard
# contains meaningful fraud investigation cases.

fraud_sample = fraud_data.copy()


deployment_data = pd.concat(
    [
        legitimate_sample,
        fraud_sample,
    ],
    ignore_index=True,
)


# ============================================================
# SHUFFLE DATASET
# ============================================================

print("\n[4/5] Shuffling deployment dataset...")

deployment_data = deployment_data.sample(
    frac=1,
    random_state=42,
).reset_index(
    drop=True
)


# ============================================================
# SAVE DATASET
# ============================================================

print("\n[5/5] Saving deployment dataset...")

deployment_data.to_csv(
    OUTPUT_PATH,
    index=False,
)


output_size_mb = (
    os.path.getsize(OUTPUT_PATH)
    / (1024 * 1024)
)

fraud_count = int(
    deployment_data["Class"].sum()
)

legitimate_count = (
    len(deployment_data)
    - fraud_count
)

fraud_percentage = (
    fraud_count
    / len(deployment_data)
) * 100


print()
print("=" * 65)
print("DEPLOYMENT DATASET CREATED")
print("=" * 65)

print(
    f"Transactions     : "
    f"{len(deployment_data):,}"
)

print(
    f"Legitimate       : "
    f"{legitimate_count:,}"
)

print(
    f"Fraud            : "
    f"{fraud_count:,}"
)

print(
    f"Fraud Percentage : "
    f"{fraud_percentage:.4f}%"
)

print(
    f"Dataset Size     : "
    f"{output_size_mb:.2f} MB"
)

print(
    f"Dataset Path     : "
    f"{OUTPUT_PATH}"
)

print()
print(
    "FraudShield deployment dataset is ready."
)

print("=" * 65)