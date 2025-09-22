# glister_select_coreset_static.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === 需要能导入你上传的 GLISTER 代码包 ===
from models.logistic_regression import LogisticRegNet
from models.set_function_onestep import SetFunctionBatch

def getFeatures(tabular_data):
  cat_list = list(tabular_data.select_dtypes(exclude=np.number))
  num_list = list(tabular_data.select_dtypes(include=np.number))

  cat_list_no_label = [feature for feature in cat_list]
  num_list_no_label = [feature for feature in num_list]

  return cat_list, num_list

def select_coreset_static(
    csv_path: str,
    label_col: str,
    budget_ratio: float = 0.2,     # 选取 20% 作为 coreset，可按需修改
    val_ratio: float = 0.2,        # 训练内部切 20% 作为验证集
    random_state: int = 42,
    standardize: bool = True,
    eta: float = 0.05,             # GLISTER one-step 的步长，常见 1e-2 ~ 1e-1
    warmup_epochs: int = 0,        # 可选：先在全训练上预热几步，使 theta_init 更稳
    lr: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_prefix: str = "glister_coreset"
):
    """
    从仅有训练集的 CSV 中一次性选择 coreset, 并返回被选行的“原始 CSV 行号”。
    """
    # 1) 读数据
    df = pd.read_csv(csv_path)

    # # remove first line
    # if df.columns[0] == df.iloc[0, 0]:
    #   df = df.drop(index=0).reset_index(drop=True)


    assert label_col in df.columns, f"Label column '{label_col}' not found."

    # 记录原始行号（用于回溯选择结果到原 CSV）
    df["_orig_idx_"] = np.arange(len(df))

    # 拆分 X/y
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # get attributes name, exclude label
    attribute_names = list(X.columns)

    # preprocess feature data
    cat_features, numeric_features = getFeatures(X)

    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
    scaler = StandardScaler()
    X_processed = X_encoded.copy()

    if len(numeric_features) > 0:
      X_processed[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])

    X = X_processed.to_numpy(dtype=np.float32)

    # change labels to index from 0 with LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    


    # 2) 在训练集内部划分验证集（GLISTER 目标需要）
    #   注意：这里的“训练集”就是你的原始 CSV；我们只是内部再切一份 val 来打分
    X_trn, X_val, y_trn, y_val, idx_trn, idx_val = train_test_split(
        X, y, df["_orig_idx_"].to_numpy(),
        test_size=val_ratio,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # 3) 标准化（常见做法，按需关闭）
    if standardize:
        scaler = StandardScaler().fit(X_trn)
        X_trn = scaler.transform(X_trn).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

    # 4) 转 torch
    X_trn_t = torch.from_numpy(X_trn).to(device)
    y_trn_t = torch.from_numpy(y_trn.astype(np.int64)).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    # 5) 定义模型与损失
    num_features = X_trn.shape[1]
    num_classes = int(np.max(y) + 1) if np.issubdtype(y.dtype, np.integer) else len(np.unique(y))
    model = LogisticRegNet(num_features, num_classes).to(device)

    loss_mean = nn.CrossEntropyLoss(reduction="mean")
    loss_nored = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 6) 可选：warmup（让 theta_init 更贴近任务，提升 GLISTER 估计稳定性）
    if warmup_epochs > 0:
        model.train()
        for _ in range(warmup_epochs):
            optimizer.zero_grad()
            logits = model(X_trn_t)
            loss = loss_mean(logits, y_trn_t)
            loss.backward()
            optimizer.step()

    theta_init = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 7) 构造 GLISTER 的集合函数对象（一次性选择）
    setfun = SetFunctionBatch(
        X_trn=X_trn_t, Y_trn=y_trn_t,
        X_val=X_val_t, Y_val=y_val_t,
        model=model,
        loss_criterion=loss_mean,
        loss_nored=loss_nored,
        eta=eta
    )

    # 8) 选择 coreset
    budget = int(max(1, round(budget_ratio * len(X_trn))))
    # 更快的懒惰贪心（推荐），也可用 naive_greedy_max
    selected_local_idx, _ = setfun.lazy_greedy_max_2(budget, theta_init)
    selected_local_idx = np.array(selected_local_idx, dtype=int)

    # 9) 将“本地训练子集索引”映射回“原 CSV 行号”
    selected_orig_idx = idx_trn[selected_local_idx]

    # 10) 导出结果
    coreset_df = df[df["_orig_idx_"].isin(selected_orig_idx)].drop(columns=["_orig_idx_"])
    idx_path = f"{save_prefix}_indices.json"
    csv_path_out = f"{save_prefix}_subset.csv"

    with open(idx_path, "w") as f:
        json.dump(selected_orig_idx.tolist(), f)

    coreset_df.to_csv(csv_path_out, index=False)

    # 返回：被选原始行号、coreset 子表、及基本统计
    return {
        "selected_orig_indices": selected_orig_idx,
        "coreset_csv": csv_path_out,
        "indices_json": idx_path,
        "budget": budget,
        "n_train_pool": len(X_trn),
        "n_val": len(X_val),
        "val_ratio": val_ratio,
    }

if __name__ == "__main__":
    # === 使用示例 ===
    result = select_coreset_static(
      csv_path="/home/xhanbh/Workspace/Research/REPO-CR4ML-ICDE26/CR4ML-ICDE26/Artifacts/data/adult/repaired/adult_dirty.csv",   # TODO: 改成你的训练 CSV
      label_col="income",                # TODO: 改成你的标签列名
      budget_ratio=0.2,                 # 选 20%
      val_ratio=0.2,                    # 内部分 20% 做验证
      random_state=42,
      standardize=True,
      eta=0.05,
      warmup_epochs=0,                  # 如需更稳的 theta_init，可调大 3~5
      lr=0.1,
      device="cuda" if torch.cuda.is_available() else "cpu",
      save_prefix="glister_core_static"
  )
    print(result)