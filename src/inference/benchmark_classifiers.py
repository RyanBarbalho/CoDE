"""
Bateria de testes para os 3 classificadores do CoDE (linear, knn, svm).
Suporta modo local (pastas real/fake) e modo streaming (ELSA_D3 via HuggingFace).
"""

import argparse
import csv
import os
import sys
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Permitir import de módulos no diretório de inference
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code_model import VITContrastiveHF
from validate_d3 import (
    RealFakeDataset,
    validate,
    calculate_acc,
    calculate_acc_svm,
    set_seed,
    AddGaussianNoise,
    ImpulsiveNoise,
)
from elsa_streaming_loader import ELSAStreamingDataset


def get_transform():
    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # AddGaussianNoise(0., 0.1),       # Insere Ruído Gaussiano
        # ImpulsiveNoise(0.05),     # Insere Salt and Pepper
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def validate_streaming(model, dataset, batch_size=128):
    """Valida modelo sobre dataset iterável (streaming). Não usa len(loader)."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    y_true, y_pred = [], []
    total = 0
    for img, label, _ in tqdm(loader, desc=model.classificator_type):
        in_tens = img.cuda()
        preds = model(in_tens).flatten().tolist()
        y_pred.extend(preds)
        y_true.extend(label.flatten().tolist())
        total += len(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if model.classificator_type in ("linear", "knn"):
        r_acc, f_acc, acc = calculate_acc(y_true, y_pred, 0.5)
    else:
        r_acc, f_acc, acc = calculate_acc_svm(y_true, y_pred)

    return r_acc, f_acc, acc, total


def run_benchmark_local(opt):
    """Executa bateria usando RealFakeDataset (pastas locais)."""
    transform = get_transform()
    dataset = RealFakeDataset(
        opt.real_path,
        opt.fake_path,
        opt.data_mode,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )
    n_total = len(dataset)

    results = []
    for ctype in ["linear", "knn", "svm"]:
        set_seed()
        model = VITContrastiveHF(classificator_type=ctype)
        model.eval()
        model.cuda()
        r_acc, f_acc, acc = validate(model, loader, opt=opt)
        results.append({
            "classifier": ctype,
            "acc_real": r_acc * 100,
            "acc_fake": f_acc * 100,
            "acc_total": acc * 100,
        })

    return results, n_total


def run_benchmark_streaming(opt):
    """Executa bateria usando ELSA_D3 em streaming."""
    transform = get_transform()
    results = []
    n_total = 0

    for ctype in ["linear", "knn", "svm"]:
        set_seed()
        model = VITContrastiveHF(classificator_type=ctype)
        model.eval()
        model.cuda()
        dataset = ELSAStreamingDataset(
            transform=transform,
            max_samples=opt.max_samples,
            use_real_images=opt.use_real_images,
        )
        r_acc, f_acc, acc, total = validate_streaming(model, dataset, opt.batch_size)
        n_total = total
        results.append({
            "classifier": ctype,
            "acc_real": r_acc * 100,
            "acc_fake": f_acc * 100,
            "acc_total": acc * 100,
        })

    return results, n_total


def print_results(results, n_total):
    """Imprime tabela de resultados."""
    print("\n=== Bateria de testes CoDE ===")
    print(f"Dataset: {n_total} amostras")
    print("-" * 50)
    print(f"{'Classificador':<12} | {'Acc Real':<10} | {'Acc Fake':<10} | {'Acc Total':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['classifier']:<12} | {r['acc_real']:>8.2f}   | {r['acc_fake']:>8.2f}   | {r['acc_total']:>8.2f}   ")
    print("-" * 50)


def save_csv(results, path):
    """Salva resultados em CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["classifier", "acc_real", "acc_fake", "acc_total"])
        w.writeheader()
        w.writerows(results)
    print(f"Resultados salvos em: {path}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_path", type=str, default=None, help="pasta real (modo local)")
    parser.add_argument("--fake_path", type=str, default=None, help="pasta fake (modo local)")
    parser.add_argument("--data_mode", type=str, default="ours", help="wang2020 ou ours")
    parser.add_argument("--result_folder", type=str, default="./results", help="pasta de saída")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    # Modo streaming
    parser.add_argument("--streaming", action="store_true", help="usar ELSA_D3 em streaming")
    parser.add_argument("--max_samples", type=int, default=2000, help="limite de amostras (streaming)")
    parser.add_argument("--use_real_images", action="store_true", help="incluir imagens reais via URL (streaming)")

    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)

    if opt.streaming:
        if not opt.real_path and not opt.fake_path:
            results, n_total = run_benchmark_streaming(opt)
        else:
            print("Modo streaming: --real_path e --fake_path ignorados.")
            results, n_total = run_benchmark_streaming(opt)
    else:
        if not opt.real_path or not opt.fake_path:
            print("Modo local exige --real_path e --fake_path.")
            sys.exit(1)
        results, n_total = run_benchmark_local(opt)

    print_results(results, n_total)
    csv_path = os.path.join(opt.result_folder, "benchmark_results.csv")
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
