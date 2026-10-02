import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

def run_benchmark(data_path="creditcard.csv", output_path="benchmark_result.json"):
    print("=" * 60)
    print("🚀 BẮT ĐẦU BENCHMARK LIGHTGBM TRÊN CPU NODE")
    print("=" * 60)
    
    # 1. Load dataset và đo thời gian
    print(f"\n[1/5] Đang đọc dữ liệu từ: {data_path} ...")
    start_load = time.perf_counter()
    df = pd.read_csv(data_path)
    time_load_data = time.perf_counter() - start_load
    print(f"  ✓ Đã load {len(df):,} dòng, {df.shape[1]} cột trong {time_load_data:.4f}s")
    
    # Tách features và target
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Tách tập train/test (80% train, 20% test, stratify để giữ tỉ lệ mẫu gian lận)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train set: {len(X_train):,} mẫu | Test set: {len(X_test):,} mẫu")
    
    # 2. Huấn luyện LightGBM và đo thời gian
    print("\n[2/5] Đang huấn luyện LightGBM Classifier ...")
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    start_train = time.perf_counter()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    time_training = time.perf_counter() - start_train
    
    best_iteration = clf.best_iteration_ if hasattr(clf, "best_iteration_") and clf.best_iteration_ else 200
    print(f"  ✓ Hoàn thành training trong {time_training:.4f}s (Best iteration: {best_iteration})")
    
    # 3. Đánh giá model trên test set
    print("\n[3/5] Đang đánh giá model trên test set ...")
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    
    print(f"  ✓ AUC-ROC:   {auc_roc:.4f}")
    print(f"  ✓ Accuracy:  {accuracy:.4f}")
    print(f"  ✓ F1-Score:  {f1:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall:    {recall:.4f}")
    
    # 4. Đo Inference Latency (1 dòng) và Throughput (1000 dòng)
    print("\n[4/5] Đang đo inference latency và throughput ...")
    single_row = X_test.iloc[[0]]
    # Warmup
    for _ in range(10):
        _ = clf.predict_proba(single_row)
        
    # Latency: trung bình 100 lần predict 1 dòng
    n_latency_runs = 100
    start_lat = time.perf_counter()
    for _ in range(n_latency_runs):
        _ = clf.predict_proba(single_row)
    latency_1_row_ms = ((time.perf_counter() - start_lat) / n_latency_runs) * 1000.0
    
    # Throughput: đo predict 1000 dòng
    batch_1000 = X_test.iloc[:1000]
    n_tp_runs = 20
    start_tp = time.perf_counter()
    for _ in range(n_tp_runs):
        _ = clf.predict_proba(batch_1000)
    avg_batch_time = (time.perf_counter() - start_tp) / n_tp_runs
    throughput_rows_per_sec = 1000.0 / avg_batch_time
    
    print(f"  ✓ Latency (1 row):          {latency_1_row_ms:.4f} ms")
    print(f"  ✓ Throughput (1000 rows):    {throughput_rows_per_sec:.2f} rows/s ({avg_batch_time*1000:.2f} ms/batch)")
    
    # 5. Lưu kết quả ra benchmark_result.json
    results = {
        "time_load_data_seconds": round(time_load_data, 4),
        "time_training_seconds": round(time_training, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(auc_roc, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "inference_latency_1_row_ms": round(latency_1_row_ms, 4),
        "inference_throughput_1000_rows_per_sec": round(throughput_rows_per_sec, 2),
        "inference_batch_1000_time_ms": round(avg_batch_time * 1000.0, 2)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[5/5] Đã ghi kết quả vào file: {output_path}")
    
    # In bảng markdown tóm tắt
    print("\n" + "=" * 60)
    print("📊 BẢNG KẾT QUẢ BENCHMARK (Sao chép vào báo cáo/README):")
    print("=" * 60)
    print("| Metric | Kết quả |")
    print("|---|---|")
    print(f"| Thời gian load data | {results['time_load_data_seconds']} s |")
    print(f"| Thời gian training | {results['time_training_seconds']} s |")
    print(f"| Best iteration | {results['best_iteration']} |")
    print(f"| AUC-ROC | {results['auc_roc']} |")
    print(f"| Accuracy | {results['accuracy']} |")
    print(f"| F1-Score | {results['f1_score']} |")
    print(f"| Precision | {results['precision']} |")
    print(f"| Recall | {results['recall']} |")
    print(f"| Inference latency (1 row) | {results['inference_latency_1_row_ms']} ms |")
    print(f"| Inference throughput (1000 rows) | {results['inference_throughput_1000_rows_per_sec']} rows/s |")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    # Tìm file creditcard.csv ở thư mục hiện tại hoặc ~/ml-benchmark/
    default_csv = "creditcard.csv"
    if not os.path.exists(default_csv):
        expanded = os.path.expanduser("~/ml-benchmark/creditcard.csv")
        if os.path.exists(expanded):
            default_csv = expanded
            
    run_benchmark(data_path=default_csv, output_path="benchmark_result.json")
