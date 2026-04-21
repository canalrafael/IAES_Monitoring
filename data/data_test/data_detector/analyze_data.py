#!/usr/bin/env python3
"""
analyze_data.py — Analisa arquivos data_detector*.txt para validar integridade e sentido dos dados.

Verifica:
  1. Parsing correto dos blocos PMU_START/PMU_END e DET_START/DET_END
  2. Estatísticas por coluna (min, max, média, mediana, std)
  3. Distribuição de labels
  4. Timestamps crescentes e intervalos
  5. Valores anomalos (zeros, outliers extremos)
  6. Correlação entre labels e métricas (ataque vs benigno)
  7. Resultados do detector vs labels reais
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Helpers sem dependências externas
# ──────────────────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else 0

def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0: return 0
    if n % 2 == 1: return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def stdev(vals):
    if len(vals) < 2: return 0
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5

def percentile(vals, p):
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s): return s[f]
    return s[f] + (k - f) * (s[c] - s[f])

def ts_to_ms(ts_str):
    """Converte HH:MM:SS:mmm para milissegundos."""
    parts = ts_str.split(':')
    if len(parts) != 4:
        return None
    try:
        hh, mm, ss, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return hh * 3600000 + mm * 60000 + ss * 1000 + ms
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────

def parse_file(filepath):
    """Parse um arquivo com blocos PMU_START/END e DET_START/END."""
    pmu_rows = []
    det_rows = []
    
    in_pmu = False
    in_det = False
    pmu_header_seen = False
    det_header_seen = False
    
    n_pmu_blocks = 0
    n_det_blocks = 0
    parse_errors = []
    
    with open(filepath, 'r', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if line == 'PMU_START':
                in_pmu = True
                pmu_header_seen = False
                n_pmu_blocks += 1
                continue
            elif line == 'PMU_END':
                in_pmu = False
                continue
            elif line == 'DET_START':
                in_det = True
                det_header_seen = False
                n_det_blocks += 1
                continue
            elif line == 'DET_END':
                in_det = False
                continue
            
            if in_pmu:
                if not pmu_header_seen:
                    pmu_header_seen = True  # skip header
                    continue
                parts = line.split(',')
                if len(parts) == 8:
                    try:
                        row = {
                            'core_id': int(parts[0]),
                            'timestamp': parts[1],
                            'timestamp_ms': ts_to_ms(parts[1]),
                            'cpu_cycles': int(parts[2]),
                            'instructions': int(parts[3]),
                            'cache_misses': int(parts[4]),
                            'branch_misses': int(parts[5]),
                            'l2_cache_access': int(parts[6]),
                            'label': int(parts[7]),
                        }
                        pmu_rows.append(row)
                    except (ValueError, TypeError) as e:
                        parse_errors.append(f"Linha {line_num}: {e} → '{line[:80]}'")
            
            if in_det:
                if not det_header_seen:
                    det_header_seen = True
                    continue
                parts = line.split(',')
                if len(parts) == 3:
                    try:
                        row = {
                            'sample_idx': int(parts[0]),
                            'status': parts[1],
                            'probability': int(parts[2]),
                        }
                        det_rows.append(row)
                    except (ValueError, TypeError) as e:
                        parse_errors.append(f"Linha {line_num}: {e} → '{line[:80]}'")

    return pmu_rows, det_rows, n_pmu_blocks, n_det_blocks, parse_errors


# ──────────────────────────────────────────────────────────────────
# Análise
# ──────────────────────────────────────────────────────────────────

def analyze_pmu(pmu_rows, filename):
    """Analisa dados PMU e imprime relatório."""
    
    if not pmu_rows:
        print("  ⚠ Nenhum dado PMU encontrado!")
        return
    
    # --- Estatísticas gerais ---
    cols = ['cpu_cycles', 'instructions', 'cache_misses', 'branch_misses', 'l2_cache_access']
    
    print(f"\n  📊 Estatísticas por coluna ({len(pmu_rows)} amostras):")
    print(f"  {'Coluna':<20} {'Min':>12} {'Max':>12} {'Média':>14} {'Mediana':>12} {'Std':>14}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*14} {'─'*12} {'─'*14}")
    
    for col in cols:
        vals = [r[col] for r in pmu_rows]
        print(f"  {col:<20} {min(vals):>12,} {max(vals):>12,} {mean(vals):>14,.1f} {median(vals):>12,.1f} {stdev(vals):>14,.1f}")
    
    # --- Distribuição de labels ---
    label_counts = defaultdict(int)
    for r in pmu_rows:
        label_counts[r['label']] += 1
    
    print(f"\n  🏷️  Distribuição de labels:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / len(pmu_rows) * 100
        print(f"    Label {label}: {count:>8,} amostras ({pct:.1f}%)")
    
    # --- Distribuição de cores ---
    core_counts = defaultdict(int)
    for r in pmu_rows:
        core_counts[r['core_id']] += 1
    
    print(f"\n  🖥️  Cores observados:")
    for cid in sorted(core_counts.keys()):
        print(f"    Core {cid}: {core_counts[cid]:>8,} amostras")
    
    # --- Timestamps ---
    ts_vals = [r['timestamp_ms'] for r in pmu_rows if r['timestamp_ms'] is not None]
    if len(ts_vals) >= 2:
        deltas = [ts_vals[i+1] - ts_vals[i] for i in range(len(ts_vals) - 1)]
        pos_deltas = [d for d in deltas if d > 0]
        neg_deltas = [d for d in deltas if d < 0]
        zero_deltas = [d for d in deltas if d == 0]
        
        first_ts = ts_vals[0]
        last_ts = ts_vals[-1]
        duration_s = (last_ts - first_ts) / 1000
        
        print(f"\n  ⏱️  Timestamps:")
        print(f"    Primeiro: {pmu_rows[0]['timestamp']} ({first_ts} ms)")
        print(f"    Último:   {pmu_rows[-1]['timestamp']} ({last_ts} ms)")
        print(f"    Duração total: {duration_s:.1f}s ({duration_s/60:.1f} min)")
        
        if pos_deltas:
            print(f"    Intervalo entre amostras (positivos):")
            print(f"      Min: {min(pos_deltas)} ms  |  Max: {max(pos_deltas)} ms  |  Média: {mean(pos_deltas):.1f} ms  |  Mediana: {median(pos_deltas):.0f} ms")
        
        if neg_deltas:
            print(f"    ⚠  {len(neg_deltas)} deltas negativos (resets de bloco ou dessincronização)")
        if zero_deltas:
            print(f"    ⚠  {len(zero_deltas)} deltas = 0 (timestamps iguais)")
    
    # --- Verificação de zeros ---
    print(f"\n  🔍 Verificação de valores anômalos:")
    
    zero_cycles = sum(1 for r in pmu_rows if r['cpu_cycles'] == 0)
    zero_instr = sum(1 for r in pmu_rows if r['instructions'] == 0)
    zero_cache = sum(1 for r in pmu_rows if r['cache_misses'] == 0)
    zero_branch = sum(1 for r in pmu_rows if r['branch_misses'] == 0)
    zero_l2 = sum(1 for r in pmu_rows if r['l2_cache_access'] == 0)
    
    total = len(pmu_rows)
    print(f"    cpu_cycles = 0:     {zero_cycles:>6} ({zero_cycles/total*100:.1f}%)")
    print(f"    instructions = 0:   {zero_instr:>6} ({zero_instr/total*100:.1f}%)")
    print(f"    cache_misses = 0:   {zero_cache:>6} ({zero_cache/total*100:.1f}%)")
    print(f"    branch_misses = 0:  {zero_branch:>6} ({zero_branch/total*100:.1f}%)")
    print(f"    l2_cache_access = 0:{zero_l2:>6} ({zero_l2/total*100:.1f}%)")
    
    # --- IPC (Instructions Per Cycle) ---
    ipcs = [r['instructions'] / r['cpu_cycles'] for r in pmu_rows if r['cpu_cycles'] > 0]
    if ipcs:
        print(f"\n  📈 IPC (Instructions Per Cycle):")
        print(f"    Min: {min(ipcs):.4f}  |  Max: {max(ipcs):.4f}  |  Média: {mean(ipcs):.4f}  |  Std: {stdev(ipcs):.4f}")
    
    # --- MPKI (Cache Misses Per 1000 Instructions) ---
    mpkis = [r['cache_misses'] * 1000 / r['instructions'] for r in pmu_rows if r['instructions'] > 0]
    if mpkis:
        print(f"\n  📈 MPKI (Cache Misses per 1K Instructions):")
        print(f"    Min: {min(mpkis):.4f}  |  Max: {max(mpkis):.4f}  |  Média: {mean(mpkis):.4f}  |  Std: {stdev(mpkis):.4f}")
    
    # --- Comparação Ataque vs Benigno ---
    attack_rows = [r for r in pmu_rows if r['label'] == 2]
    benign_rows = [r for r in pmu_rows if r['label'] == 0]
    
    if attack_rows and benign_rows:
        print(f"\n  ⚔️  Comparação Ataque (label=2) vs Benigno (label=0):")
        print(f"  {'Métrica':<20} {'Ataque (média)':>16} {'Benigno (média)':>16} {'Razão':>10}")
        print(f"  {'─'*20} {'─'*16} {'─'*16} {'─'*10}")
        
        for col in cols:
            atk_mean = mean([r[col] for r in attack_rows])
            ben_mean = mean([r[col] for r in benign_rows])
            ratio = atk_mean / ben_mean if ben_mean > 0 else float('inf')
            print(f"  {col:<20} {atk_mean:>16,.1f} {ben_mean:>16,.1f} {ratio:>10.2f}x")
        
        # IPC comparison
        atk_ipc = mean([r['instructions']/r['cpu_cycles'] for r in attack_rows if r['cpu_cycles'] > 0])
        ben_ipc = mean([r['instructions']/r['cpu_cycles'] for r in benign_rows if r['cpu_cycles'] > 0])
        print(f"  {'IPC':<20} {atk_ipc:>16.4f} {ben_ipc:>16.4f} {atk_ipc/ben_ipc if ben_ipc > 0 else 0:>10.2f}x")


def analyze_det(det_rows, pmu_rows, filename):
    """Analisa dados do detector."""
    
    if not det_rows:
        print("\n  ⚠ Nenhum dado do detector encontrado!")
        return
    
    print(f"\n  🤖 Detector ({len(det_rows)} predições):")
    
    # Status distribution
    status_counts = defaultdict(int)
    for r in det_rows:
        status_counts[r['status']] += 1
    
    for status in sorted(status_counts.keys()):
        count = status_counts[status]
        pct = count / len(det_rows) * 100
        print(f"    {status}: {count:>8,} ({pct:.1f}%)")
    
    # Probability distribution
    probs = [r['probability'] for r in det_rows]
    print(f"\n    Probabilidade:")
    print(f"      Min: {min(probs)}  |  Max: {max(probs)}  |  Média: {mean(probs):.1f}  |  Mediana: {median(probs):.0f}")
    
    # Check if always same value (saturated)
    unique_probs = set(probs)
    if len(unique_probs) <= 3:
        print(f"    ⚠  Apenas {len(unique_probs)} valor(es) únicos de probabilidade: {sorted(unique_probs)}")
        print(f"       → Detector pode estar saturado (saída constante)")
    else:
        print(f"    ✓ {len(unique_probs)} valores únicos de probabilidade (detector variando)")
    
    # Probability histogram (simple text)
    print(f"\n    Histograma de probabilidade:")
    buckets = [0] * 10
    for p in probs:
        idx = min(p // 10, 9)
        buckets[idx] += 1
    for i, count in enumerate(buckets):
        bar = '█' * int(count / max(1, max(buckets)) * 40)
        pct = count / len(probs) * 100
        print(f"      {i*10:>3}-{(i+1)*10-1:>3}%: {bar} {count} ({pct:.1f}%)")
    
    # ── Métricas de classificação ──────────────────────────────────
    if not pmu_rows or len(pmu_rows) != len(det_rows):
        print(f"\n    ⚠ Não é possível calcular métricas: PMU ({len(pmu_rows)}) ≠ DET ({len(det_rows)})")
        return
    
    # Ground truth: label 2 = ataque, label 0 = benigno
    # Predição: ATTACK = positivo, BENIGN = negativo, WARMUP = ignorado
    tp = fp = tn = fn = 0
    skipped = 0
    
    for pmu, det in zip(pmu_rows, det_rows):
        gt_label = pmu['label']
        pred = det['status']
        
        if pred == 'WARMUP':
            skipped += 1
            continue
        
        is_attack_gt = (gt_label == 2)   # ground truth: ataque
        is_attack_pred = (pred == 'ATTACK')
        
        if is_attack_gt and is_attack_pred:
            tp += 1
        elif is_attack_gt and not is_attack_pred:
            fn += 1
        elif not is_attack_gt and is_attack_pred:
            fp += 1
        else:
            tn += 1
    
    total_eval = tp + fp + tn + fn
    if total_eval == 0:
        print(f"\n    ⚠ Sem amostras válidas para calcular métricas")
        return
    
    accuracy  = (tp + tn) / total_eval * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  📐 Métricas de Classificação (excluindo {skipped} WARMUP):")
    print(f"    Amostras avaliadas: {total_eval:,}")
    print(f"\n    Matriz de Confusão:")
    print(f"                        Predito ATTACK  Predito BENIGN")
    print(f"      Real ATAQUE (2):  {tp:>14,}  {fn:>14,}")
    print(f"      Real BENIGNO (0): {fp:>14,}  {tn:>14,}")
    print(f"\n    {'Accuracy:':<16} {accuracy:>7.2f}%")
    print(f"    {'Precision:':<16} {precision:>7.2f}%")
    print(f"    {'Recall:':<16} {recall:>7.2f}%")
    print(f"    {'F1-Score:':<16} {f1:>7.2f}%")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def analyze_file(filepath):
    """Analisa um único arquivo."""
    fname = Path(filepath).name
    
    print(f"\n{'='*70}")
    print(f"  📄 Analisando: {fname}")
    print(f"{'='*70}")
    
    pmu_rows, det_rows, n_pmu, n_det, errors = parse_file(filepath)
    
    print(f"\n  📦 Blocos encontrados: {n_pmu} PMU, {n_det} DET")
    print(f"  📊 Amostras PMU: {len(pmu_rows):,}")
    print(f"  🤖 Predições DET: {len(det_rows):,}")
    
    if errors:
        print(f"\n  ❌ {len(errors)} erros de parsing:")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... e mais {len(errors)-5} erros")
    
    analyze_pmu(pmu_rows, fname)
    analyze_det(det_rows, pmu_rows, fname)
    
    # --- Veredicto ---
    print(f"\n  {'─'*60}")
    print(f"  📋 VEREDICTO para {fname}:")
    
    issues = []
    
    if len(pmu_rows) == 0:
        issues.append("❌ Sem dados PMU")
    
    if len(pmu_rows) > 0:
        zero_pct = sum(1 for r in pmu_rows if r['cache_misses'] == 0) / len(pmu_rows) * 100
        if zero_pct > 95:
            issues.append(f"⚠  {zero_pct:.0f}% de cache_misses = 0 (VMs podem estar idle)")
    
    if det_rows:
        unique_p = len(set(r['probability'] for r in det_rows))
        if unique_p <= 2:
            issues.append(f"⚠  Detector saturado ({unique_p} valores únicos de prob)")
    
    if pmu_rows:
        attack_n = sum(1 for r in pmu_rows if r['label'] == 2)
        benign_n = sum(1 for r in pmu_rows if r['label'] == 0)
        if attack_n > 0 and benign_n > 0:
            atk_cache = mean([r['cache_misses'] for r in pmu_rows if r['label'] == 2])
            ben_cache = mean([r['cache_misses'] for r in pmu_rows if r['label'] == 0])
            if atk_cache > 0 and ben_cache > 0:
                ratio = atk_cache / ben_cache
                if 0.8 < ratio < 1.2:
                    issues.append(f"⚠  Cache misses similar entre ataque e benigno (razão {ratio:.2f}x)")
                else:
                    print(f"  ✓ Cache misses diferem entre ataque/benigno ({ratio:.2f}x)")
    
    if not issues:
        print(f"  ✅ Dados parecem válidos e consistentes!")
    else:
        for issue in issues:
            print(f"  {issue}")
    

def main():
    # Auto-descobre todos os data_detector*.txt no diretório do script
    script_dir = Path(__file__).resolve().parent
    files = sorted(script_dir.glob('data_detector*.txt'))
    
    if not files:
        print(f"⚠ Nenhum arquivo data_detector*.txt encontrado em {script_dir}")
        sys.exit(1)
    
    print(f"Encontrados {len(files)} arquivo(s): {', '.join(f.name for f in files)}")
    
    for f in files:
        analyze_file(f)
    
    print(f"\n{'='*70}")
    print(f"  ✅ Análise concluída!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
